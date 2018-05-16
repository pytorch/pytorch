#include <torch/nn/modules/linear.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>

namespace torch { namespace nn {

Linear::Linear(size_t features_in, size_t features_out)
    : in_(features_in), out_(features_out) {}

void Linear::reset() {
  weight_ = add(Var(at::CPU(at::kFloat).empty({out_, in_})), "weight");
  if (with_bias_) {
    bias_ = add(Var(at::CPU(at::kFloat).empty(out_)), "bias");
  }

  const auto stdv = 1.0 / std::sqrt(weight_.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Linear::forward(variable_list input) {
  auto x = input[0];
  if (x.ndimension() == 2 && with_bias_) {
    // Fused op is marginally faster
    AT_ASSERT(x.size(1) == weight_.size(1));
    return variable_list({at::addmm(bias_, x, weight_.t())});
  }

  auto output = x.matmul(weight_.t());
  if (with_bias_) {
    output += bias_;
  }
  return variable_list({output});
}
}} // namespace torch::nn
