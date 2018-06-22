#include <torch/optim/optimizer.h>

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <memory>

namespace torch {
namespace optim {
namespace detail {
void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    auto& grad = parameter.grad();
    if (grad.defined()) {
      grad = grad.detach();
      Variable(grad).data().zero_();
    }
  }
}
} // namespace detail
} // namespace optim
} // namespace torch
