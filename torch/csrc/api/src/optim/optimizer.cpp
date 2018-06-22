#include <torch/optim/optimizer.h>

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <memory>

namespace torch {
namespace optim {
namespace detail {
OptimizerBase::OptimizerBase(std::shared_ptr<nn::Module> model)
    : model_(model) {}

void OptimizerBase::zero_grad() {
  for (auto& p : model_->parameters()) {
    auto& grad = p->grad();
    if (grad.defined()) {
      grad = grad.detach();
      torch::autograd::as_variable_ref(grad).data().zero_();
    }
  }
}
} // namespace detail
} // namespace optim
} // namespace torch
