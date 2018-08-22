#include <torch/nn/modules/linear.h>

#include <torch/tensor.h>
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
  NoGradGuard no_grad;;
  for (auto& p : parameters()) {
    p->uniform_(-stdv, stdv);
  }
}

Tensor LinearImpl::forward(Tensor input) {
  if (input.ndimension() == 2 && options.with_bias_) {
    // Fused op is marginally faster
    AT_ASSERT(input.size(1) == weight.size(1));
    return {torch::addmm(bias, input, weight.t())};
  }

  auto output = input.matmul(weight.t());
  if (options.with_bias_) {
    output += bias;
  }
  return output;
}
} // namespace nn
} // namespace torch
