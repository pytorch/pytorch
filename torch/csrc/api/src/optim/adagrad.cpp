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

Adagrad::Adagrad(ParameterContainer&& parameters, const AdagradOptions& options_)
    : Optimizer(std::forward<ParameterContainer>(parameters)), options(options_) {

    TORCH_CHECK(options.learning_rate()>=0, "Invalid learning rate: ", options.learning_rate());
    TORCH_CHECK(options.lr_decay()>=0, "Invalid lr_decay value: ", options.lr_decay());
    TORCH_CHECK(options.weight_decay()>=0, "Invalid weight_decay value: ", options.weight_decay());
    TORCH_CHECK(options.initial_accumulator_value()>=0, "Invalid initial_accumulator_value value ", options.initial_accumulator_value());
    TORCH_CHECK(options.learning_rate()>=0, "Invalid epsilon value: ", options.eps());

    // defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
    //                     initial_accumulator_value=initial_accumulator_value)
    //     super(Adagrad, self).__init__(params, defaults)
    //
    //     for group in self.param_groups:
    //         for p in group['params']:
    //             state = self.state[p]
    //             state['step'] = 0
    //             state['sum'] = torch.full_like(p.data, initial_accumulator_value)
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
template <typename Closure>
void Adagrad::step(Closure closure) {
  // loss = None
  // if closure is not None:
  //     loss = closure()
  //
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }
    // auto grad = p.grad;
    // auto curr_state = state[p];
    // state['step'] += 1

    if (options.weight_decay() > 0) {
      NoGradGuard guard;
      p.grad() = p.grad() + options.weight_decay() * p;
    }

    buffer_at(step_buffers, i) += 1.0;
    const auto clr = options.learning_rate() /
        (1.0 + (buffer_at(step_buffers, i) - 1.0) * options.lr_decay());

    auto& sum = buffer_at(sum_buffers, i);
    sum.addcmul_(p.grad(), p.grad(), 1.0);
    const auto std = buffer_at(sum_buffers, i).sqrt().add_(1e-10);

    NoGradGuard guard;
    p.addcdiv_(p.grad(), std, -clr);
  }

  // for group in self.param_groups:
  //     for p in group['params']:
  //         if p.grad is None:
  //             continue
  //
  //         grad = p.grad.data
  //         state = self.state[p]
  //
  //         state['step'] += 1
  //
  //         if group['weight_decay'] != 0:
  //             if p.grad.data.is_sparse:
  //                 raise RuntimeError("weight_decay option is not compatible with sparse gradients")
  //             grad = grad.add(group['weight_decay'], p.data)
  //
  //         clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
  //
  //         if grad.is_sparse:
  //             grad = grad.coalesce()  # the update is non-linear so indices must be unique
  //             grad_indices = grad._indices()
  //             grad_values = grad._values()
  //             size = grad.size()
  //
  //             def make_sparse(values):
  //                 constructor = grad.new
  //                 if grad_indices.dim() == 0 or values.dim() == 0:
  //                     return constructor().resize_as_(grad)
  //                 return constructor(grad_indices, values, size)
  //             state['sum'].add_(make_sparse(grad_values.pow(2)))
  //             std = state['sum'].sparse_mask(grad)
  //             std_values = std._values().sqrt_().add_(group['eps'])
  //             p.data.add_(-clr, make_sparse(grad_values / std_values))
  //         else:
  //             state['sum'].addcmul_(1, grad, grad)
  //             std = state['sum'].sqrt().add_(group['eps'])
  //             p.data.addcdiv_(-clr, grad, std)
  //
  // return loss
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
