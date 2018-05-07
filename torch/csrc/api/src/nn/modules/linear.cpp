#include <torch/nn/modules/linear.h>

namespace torch { namespace nn {
variable_list Linear::forward(variable_list input) {
  auto x = input[0];
  if (x.ndimension() == 2 && !no_bias_) {
    // Fused op is marginally faster
    assert(x.size(1) == weight.size(1));
    return variable_list({at::addmm(bias, x, weight.t())});
  }

  auto output = x.matmul(weight.t());
  if (!no_bias_) {
    output += bias;
  }
  return variable_list({output});
}

void Linear::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(weight.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

void Linear::initialize_parameters() {
  weight = this->add(
      Var(DefaultTensor(at::kFloat).tensor({nout, nin}), true), "weight");
  if (!no_bias_) {
    bias =
        this->add(Var(DefaultTensor(at::kFloat).tensor({nout}), true), "bias");
  }
}
}} // namespace torch::nn
