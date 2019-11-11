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

AdagradOptions::AdagradOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/*
def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]

            state['step'] += 1

            if group['weight_decay'] != 0:
                if p.grad.data.is_sparse:
                    raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                grad = grad.add(group['weight_decay'], p.data)

            clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

            if grad.is_sparse:
                grad = grad.coalesce()  # the update is non-linear so indices must be unique
                grad_indices = grad._indices()
                grad_values = grad._values()
                size = grad.size()

                def make_sparse(values):
                    constructor = grad.new
                    if grad_indices.dim() == 0 or values.dim() == 0:
                        return constructor().resize_as_(grad)
                    return constructor(grad_indices, values, size)
                state['sum'].add_(make_sparse(grad_values.pow(2)))
                std = state['sum'].sparse_mask(grad)
                std_values = std._values().sqrt_().add_(group['eps'])
                p.data.add_(-clr, make_sparse(grad_values / std_values))
            else:
                state['sum'].addcmul_(1, grad, grad)
                std = state['sum'].sqrt().add_(group['eps'])
                p.data.addcdiv_(-clr, grad, std)

    return loss
*/

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();
      auto& state = state_[p.unsafeGetTensorImpl()];

      state.step_++;

      if(group.options().weight_decay() != 0) {
        TORCH_CHECK(!p.grad().data().is_sparse(), "weight_decay option is not compatible with sparse gradients");
        grad = grad.add(p.data(), group.options().weight_decay());
      }
      const auto clr = group.options().learning_rate() /
          (1 + (state.step_ - 1) * group.options().lr_decay());

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
        state.sum_.add_(make_sparse(grad_values.pow(2)));
        auto std = state.sum_.sparse_mask(grad);
        const auto std_values = std._values().sqrt_().add_(group.options().eps());

        p.data().add_(make_sparse(grad_values / std_values), -clr);
      }
      else {
        state.sum_.addcmul_(grad, grad, 1.0);
        const auto std = state.sum_.sqrt().add_(group.options().eps());
        p.data().addcdiv_(grad, std, -clr);
      }
    }
  }
}

void Adagrad::add_parameters(const std::vector<Tensor>& parameters) {
  param_groups_.push_back(AdagradParamGroup(parameters, *defaults_));
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
