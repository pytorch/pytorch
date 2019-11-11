#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <c10/util/flat_hash_map.h>

#include <ATen/ATen.h>

#include <functional>

using c10::Dict;

namespace torch {
namespace optim {

AdagradOptions convert_ivalue_to_options(at::IValue ivalue) {
    c10::Dict<at::IValue, at::IValue> dict = ivalue.toGenericDict();
    AdagradOptions options(0);
    options.learning_rate(dict.at("learning_rate").toDouble());
    options.lr_decay(dict.at("lr_decay").toDouble());
    options.weight_decay(dict.at("weight_decay").toDouble());
    options.initial_accumulator_value(dict.at("initial_accumulator_value").toDouble());
    options.eps(dict.at("eps").toDouble());
    return options;
}

AdagradOptions::AdagradOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& group : param_groups) {
    AdagradOptions options = convert_ivalue_to_options(group.at("options"));
    for (auto& p : group.at("params").toTensorListRef()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();

      state.at(p).insert_or_assign("step", state.at(p).at("step").toInt()+1);
      if(options.weight_decay() != 0) {
        TORCH_CHECK(!p.grad().data().is_sparse(), "weight_decay option is not compatible with sparse gradients");
        NoGradGuard guard;
        grad += options.weight_decay() * p.data();
      }
      const auto clr =  options.learning_rate() /
          (1.0 + (state.at(p).at("step").toInt() - 1.0) * options.lr_decay());

      if(grad.is_sparse()) {
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.sizes();

        auto make_sparse = [&] (Tensor values) -> Tensor /*confirm*/ {
          if(grad_indices.dim() == 0 || values.dim() == 0) {
            return torch::empty({0}, grad.options()).resize_as_(grad);
          }
          return torch::sparse_coo_tensor(grad_indices, values, size, grad.options());
        };
        state.at(p).at("sum").toTensor().add_(make_sparse(grad_values.pow(2)));
        auto std = state.at(p).at("sum").toTensor().sparse_mask(grad);
        const auto std_values = std.sqrt().add_(options.eps());

        p.data().add_(make_sparse(grad_values / std_values), -clr);
      }
      else {
        state.at(p).at("sum").toTensor().addcmul_(grad, grad, 1.0);
        const auto std = state.at(p).at("sum").toTensor().sqrt().add_(options.eps());
        p.data().addcdiv_(grad, std, -clr);
      }
    }
    group.insert_or_assign("options", options.convert_options_to_ivalue());
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
