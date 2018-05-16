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

at::Scalar OptimizerImpl::NoLoss() {
  return at::Scalar();
}

void OptimizerImpl::set_model(std::shared_ptr<nn::Module> model) {
  model_ = model;
}

at::Tensor LBFGS::gather_flat_grad() {
  std::vector<at::Tensor> views;
  for(auto& pair : model_->parameters()) {
    views.push_back(torch::autograd::as_variable_ref(pair.second.grad()).data().view(-1));
  }
  return at::cat(views);
}

void LBFGS::add_grad(const at::Scalar& step_size, const at::Tensor& update) {
  int offset = 0;
  for(auto& pair : model_->parameters()) {
    int numel = pair.second.numel();
    at::Tensor& pd = pair.second.data();
    pd.add_(update.slice(0, offset, offset + numel, 1).view_as(pd), step_size);
    offset += numel;
  }
}

at::Scalar LBFGS::step(std::function<at::Scalar()> closure) {
  at::Scalar orig_loss = closure();
  at::Scalar loss = orig_loss;
  int current_evals = 1;
  func_evals += 1;

  at::Tensor flat_grad = gather_flat_grad();
  at::Scalar abs_grad_sum = at::Scalar(flat_grad.abs().sum());

  if(at::Scalar(abs_grad_sum).toFloat() <= tolerance_grad_) {
    return loss;
  }

  at::Tensor ONE = flat_grad.type().scalarTensor(1);

  int n_iter = 0;
  while(n_iter < max_iter_) {
    n_iter++;
    state_n_iter++;

    if(state_n_iter == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
      prev_flat_grad = flat_grad.clone();
    } else {
      at::Tensor y = flat_grad.sub(prev_flat_grad);
      at::Tensor s = d.mul(t);
      at::Scalar ys = at::Scalar(y.dot(s));

      if(ys.toFloat() > 1e-10) {
        // updating memory

        if((int)old_dirs.size() == history_size_) {
          // shift history by one (limited memory)
          old_dirs.pop_front();
          old_stps.pop_front();
        }

        // store new direction/step
        old_dirs.push_back(y);
        old_stps.push_back(s);

        // update scale of initial Hessian approximation
        H_diag = ys / y.dot(y);
      }

      int num_old = old_dirs.size();

      for(int i = 0; i < num_old; i++) {
        ro[i] = ONE / old_dirs[i].dot(old_stps[i]);
      }

      at::Tensor q = flat_grad.neg();
      for(int i = num_old - 1; i >= 0; i--) {
        al[i] = old_stps[i].dot(q) * ro[i];
        q.add_(old_dirs[i], at::Scalar(-al[i]));
      }

      // Multiply by initial Hessian
      // r/d is the final direction
      at::Tensor r = q.mul(H_diag);
      d = r;

      for(int i = 0; i < num_old; i++) {
        at::Tensor be_i = old_dirs[i].dot(r) * ro[i];
        r.add_(old_stps[i], at::Scalar(al[i] - be_i));
      }
      prev_flat_grad.copy_(flat_grad);
    }

    /**
     * comute step length
     */

    // reset initial guess for step size
    if(n_iter == 1) {
      t = at::Scalar(at::min(ONE, ONE / abs_grad_sum) * lr_);
    } else {
      t = lr_;
    }

    at::Scalar gtd = at::Scalar(flat_grad.dot(d));
    add_grad(t, d);
    int ls_func_evals = 0;
    if(n_iter != max_iter_) {
      // re-evaluate function only if not in last iteration
      // the reason we do this: in a stochastic setting,
      // no use to re-evaluate that function here
      loss = closure();
      flat_grad = gather_flat_grad();
      abs_grad_sum = at::Scalar(flat_grad.abs().sum());
      ls_func_evals = 1;
    }

    current_evals += ls_func_evals;

    /**
     * Check conditions
     */
    
    if(n_iter == max_iter_) {
      break;
    } else if (current_evals >= max_eval_) {
      break;
    } else if (abs_grad_sum.toFloat() <= tolerance_grad_) {
      break;
    } else if (gtd.toFloat() > -tolerance_grad_) {
      break;
    } else if (at::Scalar(d.mul(t).abs_().sum()).toFloat() <= tolerance_change_) {
      break;
    } else if (std::abs(loss.toFloat() - prev_loss.toFloat()) < tolerance_change_) {
      break;
    } 
  }
  return orig_loss;
}

void LBFGS::init_state() {
  d = at::CPU(at::kFloat).empty({0});
  t = 0;
  H_diag = at::CPU(at::kFloat).empty({0});
  prev_flat_grad = at::CPU(at::kFloat).empty({0});
  prev_loss = 0;
  ro.resize(history_size_);
  al.resize(history_size_);
  func_evals = 0;
  state_n_iter = 0;
}

at::Scalar SGD::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
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
  return loss;
}

void SGD::init_state() {
  momentum_buffers_.clear();
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
at::Scalar Adagrad::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
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
  return loss;
}

void Adagrad::init_state() {
  sum_.clear();
  step_.clear();
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
at::Scalar RMSprop::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
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
  return loss;
}

void RMSprop::init_state() {
  square_avg_buffer_.clear();
  momentum_buffer_.clear();
  grad_avg_buffer_.clear();
}

at::Scalar Adam::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
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
  return loss;
}

void Adam::init_state() {
  step_buffer_.clear();
  exp_avg_buffer_.clear();
  exp_avg_sq_buffer_.clear();
}

} // namespace torch
