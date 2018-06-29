#include <torch/nn/modules/linear.h>

#include <torch/tensor.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {
LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

LinearImpl::LinearImpl(LinearOptions options) : options_(std::move(options)) {
  reset();
}

void LinearImpl::reset() {
  weight_ =
      register_parameter("weight", torch::empty({options_.out_, options_.in_}));
  if (options_.with_bias_) {
    bias_ = register_parameter("bias", torch::empty(options_.out_));
  }

  const auto stdv = 1.0 / std::sqrt(weight_.size(1));
  for (auto& p : parameters()) {
    p->data().uniform_(-stdv, stdv);
  }
}

Tensor LinearImpl::forward(Tensor input) {
  if (input.ndimension() == 2 && options_.with_bias_) {
    // Fused op is marginally faster
    AT_ASSERT(input.size(1) == weight_.size(1));
    return {torch::addmm(bias_, input, weight_.t())};
  }

  auto output = input.matmul(weight_.t());
  if (options_.with_bias_) {
    output += bias_;
  }
  return output;
}

const LinearOptions& LinearImpl::options() const noexcept {
  return options_;
}
} // namespace nn
} // namespace torch
