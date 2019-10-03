#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <ATen/core/functional.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::Variable;

torch::autograd::variable_list RecvRpcBackward::apply(
    torch::autograd::variable_list&& grads) {
  std::vector<Variable> outputGrads;
  for (const auto& grad : grads) {
    if (grad.defined()) {
      outputGrads.emplace_back(grad);
    } else {
      outputGrads.emplace_back(at::zeros_like(grad));
    }
  }

  return outputGrads;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
