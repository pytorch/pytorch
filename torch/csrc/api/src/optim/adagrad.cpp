#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>
#include <torch/optim/serialize.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

AdagradOptions::AdagradOptions(double lr) : lr_(lr) {}

bool operator==(const AdagradOptions& lhs, const AdagradOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
          (lhs.lr_decay() == rhs.lr_decay()) &&
          (lhs.weight_decay() == rhs.weight_decay()) &&
          (lhs.initial_accumulator_value() == rhs.initial_accumulator_value()) &&
          (lhs.eps() == rhs.eps());
}

void AdagradOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(initial_accumulator_value);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
}

void AdagradOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, initial_accumulator_value);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
}

bool operator==(const AdagradParamState& lhs, const AdagradParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
            torch::equal(lhs.sum(), rhs.sum());
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
Tensor Adagrad::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad();
      TORCH_INTERNAL_ASSERT(state_[c10::guts::to_string(p.unsafeGetTensorImpl())] != nullptr, "state found NULL for the Tensor ", p);
      auto& state = static_cast<AdagradParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto& options = static_cast<AdagradOptions&>(group.options());

      state.step(state.step() + 1);

      if (options.weight_decay() != 0) {
        TORCH_CHECK(!p.grad().is_sparse(), "weight_decay option is not compatible with sparse gradients");
        grad = grad.add(p, options.weight_decay());
      }
      const auto clr = options.lr() /
          (1 + static_cast<double>(state.step() - 1) * options.lr_decay());

      if (grad.is_sparse()) {
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.sizes();

        auto make_sparse = [&] (const Tensor& values) -> Tensor {
          if (grad_indices.dim() == 0 || values.dim() == 0) {
            return torch::empty({0}, grad.options()).resize_as_(grad);
          }
          return torch::sparse_coo_tensor(grad_indices, values, size, grad.options());
        };
        state.sum(state.sum().add_(make_sparse(grad_values.pow(2))));
        auto std = state.sum().sparse_mask(grad);
        const auto std_values = std._values().sqrt_().add_(options.eps());

        p.add_(make_sparse(grad_values / std_values), -clr);
      }
      else {
        state.sum(state.sum().addcmul_(grad, grad, 1.0));
        const auto std = state.sum().sqrt().add_(options.eps());
        p.addcdiv_(grad, std, -clr);
      }
    }
  }
  return loss;
}

void Adagrad::add_parameters(const std::vector<Tensor>& parameters) {
  return _add_parameters_new_design(parameters);
}

const std::vector<Tensor>& Adagrad::parameters() const noexcept {
  return _parameters_new_design();
}

std::vector<Tensor>& Adagrad::parameters() noexcept {
  return _parameters_new_design();
}

size_t Adagrad::size() const noexcept {
  return _size_new_design();
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  }
  else { // deserializing archives saved in old format (prior to version 1.5.0)
    TORCH_WARN(
      "Your serialized Adagrad optimizer is still using the old serialization format. "
      "You should re-save your Adagrad optimizer to use the new serialization format.");
    std::vector<Tensor> sum_buffers;
    std::vector<int64_t> step_buffers;
    torch::optim::serialize(archive, "sum_buffers", sum_buffers);
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (size_t idx = 0; idx < params.size(); idx++) {
      auto state = std::make_unique<AdagradParamState>();
      state->step(step_buffers[idx]);
      state->sum(sum_buffers[idx]);
      state_[c10::guts::to_string(params[idx].unsafeGetTensorImpl())] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
