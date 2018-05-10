#include <torch/nn/modules/dropout.h>

namespace torch { namespace nn {

Dropout::Dropout(double p) : p_(p) {
  assert(p < 1 && p >= 0);
}

variable_list Dropout::forward(variable_list inputs) {
  if (p_ == 0 || !is_training())
    return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor(x.sizes());
    noise = (noise.uniform_(0, 1) > p_)
                .toType(x.type().scalarType())
                .mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}

Dropout2d::Dropout2d(double p) : p_(p) {
  assert(p < 1 && p >= 0);
}

variable_list Dropout2d::forward(variable_list inputs) {
  if (p_ == 0 || !is_training())
    return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor({x.size(0), x.size(1), 1, 1});
    noise = (noise.uniform_(0, 1) > p_)
                .toType(x.type().scalarType())
                .mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}
}} // namespace torch::nn
