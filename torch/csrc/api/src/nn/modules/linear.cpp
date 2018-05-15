#include <torch/nn/modules/linear.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>

namespace torch { namespace nn {

Linear::Linear(uint32_t nin, uint32_t nout, bool with_bias)
    : nin(nin),
      nout(nout),
      weight(add(Var(at::CPU(at::kFloat).empty({nout, nin})), "weight")) {
  if (with_bias) {
    bias.emplace(add(Var(at::CPU(at::kFloat).empty(nout)), "bias"));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Linear::forward(variable_list input) {
  auto x = input[0];
  if (x.ndimension() == 2 && bias.has_value()) {
    // Fused op is marginally faster
    assert(x.size(1) == weight.size(1));
    return variable_list({at::addmm(*bias, x, weight.t())});
  }

  auto output = x.matmul(weight.t());
  if (bias.has_value()) {
    output += *bias;
  }
  return variable_list({output});
}
}} // namespace torch::nn
