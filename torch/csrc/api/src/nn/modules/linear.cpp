#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {

LinearImpl::LinearImpl(LinearOptions options) : options(options) {
  reset();
}

void LinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out_, options.in_}));
  if (options.with_bias_) {
    bias = register_parameter("bias", torch::empty(options.out_));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Linear(in=" << options.in_
         << ", out=" << options.out_ << ", with_bias=" << options.with_bias_
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  AT_ASSERT(!options.with_bias_ || bias.defined());
  return torch::linear(input, weight, bias);
}
} // namespace nn
} // namespace torch
