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

void AdagradOptions::serialize(torch::serialize::InputArchive& archive) {
  c10::IValue ivalue;
  archive.read("learning_rate", ivalue);
  this->learning_rate(ivalue.toDouble());
  archive.read("lr_decay", ivalue);
  this->lr_decay(ivalue.toDouble());
  archive.read("weight_decay", ivalue);
  this->weight_decay(ivalue.toDouble());
  archive.read("initial_accumulator_value", ivalue);
  this->initial_accumulator_value(ivalue.toDouble());
  archive.read("eps", ivalue);
  this->eps(ivalue.toDouble());
}

void AdagradOptions::serialize(torch::serialize::OutputArchive& archive) const {
  archive.write("learning_rate", IValue(this->learning_rate()));
  archive.write("lr_decay", IValue(this->lr_decay()));
  archive.write("weight_decay", IValue(this->weight_decay()));
  archive.write("initial_accumulator_value", IValue(this->initial_accumulator_value()));
  archive.write("eps", IValue(this->eps()));
}

void AdagradParamState::serialize(torch::serialize::InputArchive& archive) {
    c10::IValue step_, sum_;
    archive.read("step", step_);
    archive.read("sum", sum_);
    this->step(step_.toInt());
    this->sum(sum_.toTensor());
}

void AdagradParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", IValue(this->step()));
    archive.write("sum", IValue(this->sum()));
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();
      // TODO: assert that `state_[p.unsafeGetTensorImpl()]` exists and is not a null pointer, before dereferencing it
      TORCH_CHECK(state_[c10::guts::to_string(p.unsafeGetTensorImpl())] != NULL, "state found NULL for the Tensor ", p);
      auto& state = static_cast<AdagradParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto& options = static_cast<AdagradOptions&>(group.options());

      state.step(state.step() + 1);

      if(options.weight_decay() != 0) {
        TORCH_CHECK(!p.grad().data().is_sparse(), "weight_decay option is not compatible with sparse gradients");
        grad = grad.add(p.data(), options.weight_decay());
      }
      const auto clr = options.learning_rate() /
          (1 + (state.step() - 1) * options.lr_decay());

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
        state.sum(state.sum().add_(make_sparse(grad_values.pow(2))));
        auto std = state.sum().sparse_mask(grad);
        const auto std_values = std._values().sqrt_().add_(options.eps());

        p.data().add_(make_sparse(grad_values / std_values), -clr);
      }
      else {
        state.sum(state.sum().addcmul_(grad, grad, 1.0));
        const auto std = state.sum().sqrt().add_(options.eps());
        p.data().addcdiv_(grad, std, -clr);
      }
    }
  }
}

void Adagrad::add_parameters(const std::vector<Tensor>& parameters) {
  param_groups_.push_back(OptimizerParamGroup(parameters, defaults_->clone()));
}

const std::vector<Tensor>& Adagrad::parameters() const noexcept {
  return param_groups_[0].params();
}

std::vector<Tensor>& Adagrad::parameters() noexcept {
  return param_groups_[0].params();
}

size_t Adagrad::size() const noexcept {
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
