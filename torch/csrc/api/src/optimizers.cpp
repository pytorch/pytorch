#include "torch/optimizers.h"

#include <torch/nn/module.h>

namespace torch {

void OptimizerImpl::zero_grad() {
  for (auto p : model_->parameters()) {
    auto& grad = p.second.grad();
    if (grad.defined()) {
      grad = grad.detach();
      torch::autograd::as_variable_ref(grad).data().zero_();
    }
  }
}

void OptimizerImpl::set_model(std::shared_ptr<nn::Module> model) {
  model_ = model;
}

void SGD::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined())
      continue;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };

    if (momentum_ != 0) {
      at::Tensor buf;
      if (momentum_buffers_.find(name) == momentum_buffers_.end()) {
        buf = momentum_buffers_[name] = at::zeros_like(p);
        buf.mul_(momentum_).add_(d_p);
      } else {
        buf = momentum_buffers_[name];
        buf.mul_(momentum_).add_(d_p, 1 - dampening_);
      }

      if (nesterov_) {
        d_p = d_p.add(buf, momentum_);
      } else {
        d_p = buf;
      }
    }

    p.add_(d_p, -lr_);
  }
}

void SGD::init_state() {
  momentum_buffers_.clear();
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined())
      continue;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };
    auto& step = step_[name];
    step += 1.0;
    auto clr = lr_ / (1.0 + (step - 1.0) * lr_decay_);
    at::Tensor buf;
    if (sum_.find(name) == sum_.end()) {
      buf = sum_[name] = at::zeros_like(p);
    } else {
      buf = sum_[name];
    }

    buf.addcmul_(d_p, d_p, 1.0);
    at::Tensor std = buf.sqrt().add_(1e-10);
    p.addcdiv_(d_p, std, -clr);
  }
}

void Adagrad::init_state() {
  sum_.clear();
  step_.clear();
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
void RMSprop::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined())
      continue;

    if (square_avg_buffer_.find(name) == square_avg_buffer_.end()) {
      square_avg_buffer_[name] = at::zeros_like(p);
      if (momentum_) {
        momentum_buffer_[name] = at::zeros_like(p);
      };
      if (centered_) {
        grad_avg_buffer_[name] = at::zeros_like(p);
      };
    };

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };

    auto& square_avg = square_avg_buffer_[name];
    square_avg.mul_(alpha_).addcmul_(d_p, d_p, 1.0 - alpha_);

    at::Tensor avg;
    if (centered_) {
      auto& grad_avg = grad_avg_buffer_[name];
      grad_avg.mul_(alpha_).add_(d_p, 1.0 - alpha_);
      avg = square_avg.addcmul(grad_avg, grad_avg, -1.0).sqrt().add_(eps_);
    } else {
      avg = square_avg.sqrt().add_(eps_);
    };

    if (momentum_ > 0) {
      auto& buf = momentum_buffer_[name];
      buf.mul_(momentum_).addcdiv_(d_p, avg);
      p.add_(buf, -lr_);
    } else {
      p.addcdiv_(d_p, avg, -lr_);
    };
  }
}

void RMSprop::init_state() {
  square_avg_buffer_.clear();
  momentum_buffer_.clear();
  grad_avg_buffer_.clear();
}

void Adam::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined())
      continue;

    if (step_buffer_.find(name) == step_buffer_.end()) {
      step_buffer_[name] = 0;
      exp_avg_buffer_[name] = at::zeros_like(p);
      exp_avg_sq_buffer_[name] = at::zeros_like(p);
      if (amsgrad_) {
        max_exp_avg_sq_buffer_[name] = at::zeros_like(p);
      };
    }

    auto& step = step_buffer_[name];
    auto& exp_avg = exp_avg_buffer_[name];
    auto& exp_avg_sq = exp_avg_sq_buffer_[name];

    step += 1;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    }

    exp_avg.mul_(beta1_).add_(d_p, 1 - beta1_);
    exp_avg_sq.mul_(beta2_).addcmul_(d_p, d_p, 1 - beta2_);

    at::Tensor denom;
    if (amsgrad_) {
      auto& max_exp_avg_sq = max_exp_avg_sq_buffer_[name];
      at::max_out(max_exp_avg_sq, max_exp_avg_sq, exp_avg_sq);
      denom = max_exp_avg_sq.sqrt().add_(eps_);
    } else {
      denom = exp_avg_sq.sqrt().add_(eps_);
    };

    auto bias_correction1 = 1 - std::pow(beta1_, step);
    auto bias_correction2 = 1 - std::pow(beta2_, step);
    auto step_size = lr_ * std::sqrt(bias_correction2) / bias_correction1;

    p.addcdiv_(exp_avg, denom, -step_size);
  }
}

void Adam::init_state() {
  step_buffer_.clear();
  exp_avg_buffer_.clear();
  exp_avg_sq_buffer_.clear();
}

} // namespace torch
