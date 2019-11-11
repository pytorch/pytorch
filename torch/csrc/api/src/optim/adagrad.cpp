#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

AdagradOptions::AdagradOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& group : param_groups) {
    for (auto p : group.at("params").toTensorList()) {
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

      auto& sum = state_.at("sum").toTensor();

      if(grad.is_sparse()) {
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.size();

        [&] make_sparse(Tensor values) -> Tensor /*confirm*/ {
          auto constructor = grad.new(); //confirm
          if(grad_indices.dim() == 0 || values.dim() == 0) {
            return constructor().resize_as_(grad);
          }
          return constructor(grad_indices, values, size);
        }
        sum.add_(make_sparse(grad_values.pow(2)));
        auto std = state_.at("sum").toTensor().sparse_mask(grad);
        const auto std_values = std.sqrt().add_(group.at("options").eps());
        p.data().add_(-clr, make_sparse(grad_values / std_values));
      }
      else {
        sum.addcmul_(grad, grad, 1.0);
        const auto std = state_.at("sum").toTensor().sqrt().add_(group.at("options").eps());
        p.data().addcdiv_(-clr, grad, std);
      }
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
