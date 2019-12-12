#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>
#include <torch/optim/serialize.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

AdagradOptions::AdagradOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

bool operator==(const AdagradOptions& lhs, const AdagradOptions& rhs) {
  auto eq = (lhs.learning_rate() == rhs.learning_rate()) &&
            (lhs.lr_decay() == rhs.lr_decay()) &&
            (lhs.weight_decay() == rhs.weight_decay()) &&
            (lhs.initial_accumulator_value() == rhs.initial_accumulator_value()) &&
            (lhs.eps() == rhs.eps());
  return eq;
}

void AdagradOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(learning_rate);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(initial_accumulator_value);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
}

void AdagradOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, learning_rate);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, initial_accumulator_value);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
}

bool operator==(const AdagradParamState& lhs, const AdagradParamState& rhs) {
  auto eq = (lhs.step() == rhs.step()) &&
            torch::equal(lhs.sum(), rhs.sum());
  return eq;
}

void AdagradParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(sum);
}

void AdagradParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, sum);
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
  if(archive.keys().size() == 3){
    serialize(*this, archive);
  }
  else {
    std::vector<Tensor> sum_buffers;
    std::vector<int64_t> step_buffers;
    torch::optim::serialize(archive, "sum_buffers", sum_buffers);
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    // since there were no param_groups before, assuming all tensors are now in one param_group
    std::vector<Tensor> params = param_groups_[0].params();
    for(size_t idx = 0; idx<params.size(); idx++) {
      auto state = c10::guts::make_unique<AdagradParamState>();
      state->step(step_buffers[idx]);
      state->sum(sum_buffers[idx]);
      state_[c10::guts::to_string(params[idx].unsafeGetTensorImpl())] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
