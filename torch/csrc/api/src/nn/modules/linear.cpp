#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {
LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

LinearImpl::LinearImpl(LinearOptions options) : options(std::move(options)) {
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

Tensor LinearImpl::forward(Tensor input) {
  AT_ASSERT(!options.with_bias_ || bias.defined());
  return torch::linear(input, weight, bias);
}
} // namespace nn
} // namespace torch
