#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <ATen/core/functional.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::Variable;

torch::autograd::variable_list RecvRpcBackward::apply(
    torch::autograd::variable_list&& grads) {
  // Get size and options for tensors.
  at::IntArrayRef sizes;
  at::TensorOptions o;
  for (auto v : grads) {
    if (v.defined()) {
      sizes = v.sizes();
      o = static_cast<at::Tensor>(v).options();
      break;
    }
  }

  // Put in zeros for grads that are not defined.
  auto output_grads = c10::fmap(grads, [&](const Variable& v) {
    return (v.defined() ? v : Variable(at::zeros({}, o).expand(sizes)));
  });

  return output_grads;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
