#include <torch/optim/optimizer.h>

#include <torch/tensor.h>

#include <ATen/Error.h>

namespace torch {
namespace optim {
namespace detail {
void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    auto& grad = parameter.grad();
    if (grad.defined()) {
      grad = grad.detach();
      Tensor(grad).data().zero_();
    }
  }
}
} // namespace detail
} // namespace optim
} // namespace torch
