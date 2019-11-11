#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

AdagraddefaultOptions::AdagraddefaultOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto group : param_groups) {
    for (auto p : group.at("params")) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();
      auto& state_ = state.at(p.unsafeGetTensorImpl());
      state_.at("step").toInt()+=1;

      if(group.at("options").weight_decay() != 0) {
        TORCH_CHECK(!p.grad().data().is_sparse(), "weight_decay option is not compatible with sparse gradients");

        NoGradGuard guard;
        grad += group.at("options").weight_decay() * p.data();
      }

      const auto clr =  group.at("options").learning_rate() /
          (1.0 + (state_.at("step").toInt() - 1.0) * group.at("options").lr_decay());

      if(grad.is_sparse()) {
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.size();
        //add a lambda fn
        //ad makesparse fn
      }
      else {
        //fill up
      }
      auto& sum = state_.at("sum").toTensor();
      sum.addcmul_(grad, grad, 1.0);
      const auto std = state_.at("sum").toTensor().sqrt().add_(1e-10);
      NoGradGuard guard;
      p.addcdiv_(grad, std, -clr);
    }
  }
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
