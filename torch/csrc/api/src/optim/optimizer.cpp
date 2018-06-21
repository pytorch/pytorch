#include <torch/optim/optimizer.h>

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <memory>

namespace torch {
namespace optim {
void OptimizerImpl::zero_grad() {
  for (auto& p : model_->parameters()) {
    auto& grad = p->grad();
    if (grad.defined()) {
      grad = grad.detach();
      torch::autograd::as_variable_ref(grad).data().zero_();
    }
  }
}

at::Scalar OptimizerImpl::NoLoss() {
  return at::Scalar();
}

void OptimizerImpl::set_model(std::shared_ptr<nn::Module> model) {
  model_ = model;
}

} // namespace optim
} // namespace torch
