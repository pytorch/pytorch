#include <torch/optim/lbfgs.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>
#include <vector>

namespace torch {
namespace optim {

LBFGSOptions::LBFGSOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

Tensor LBFGS::gather_flat_grad() {
  std::vector<Tensor> views;
  for (auto& parameter : parameters_) {
    views.push_back(autograd::Variable(parameter.grad()).data().view(-1));
  }
  return at::cat(views);
}

void LBFGS::add_grad(const torch::Scalar& step_size, const Tensor& update) {
  int64_t offset = 0;
  for (auto& parameter : parameters_) {
    int64_t numel = parameter.numel();
    Tensor& pd = autograd::Variable(parameter).data();
    pd.add_(update.slice(0, offset, offset + numel, 1).view_as(pd), step_size);
    offset += numel;
  }
}

torch::Tensor LBFGS::step(LossClosure closure) {
  torch::Tensor orig_loss = closure();
  torch::Tensor loss = orig_loss.clone();
  int64_t current_evals = 1;
  func_evals += 1;

  Tensor flat_grad = gather_flat_grad();
  torch::Scalar abs_grad_sum = torch::Scalar(flat_grad.abs().sum());

  if (torch::Scalar(abs_grad_sum).toFloat() <= options.tolerance_grad_) {
    return loss;
  }

  Tensor ONE = flat_grad.type().scalarTensor(1);

  int64_t n_iter = 0;
  while (n_iter < options.max_iter_) {
    n_iter++;
    state_n_iter++;

    if (state_n_iter == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
      prev_flat_grad = flat_grad.clone();
    } else {
      Tensor y = flat_grad.sub(prev_flat_grad);
      Tensor s = d.mul(t);
      torch::Scalar ys = torch::Scalar(y.dot(s));

      if (ys.toFloat() > 1e-10) {
        // updating memory

        if (old_dirs.size() == options.history_size_) {
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

      int64_t num_old = old_dirs.size();

      for (int64_t i = 0; i < num_old; i++) {
        ro.at(i) = ONE / old_dirs.at(i).dot(old_stps.at(i));
      }

      Tensor q = flat_grad.neg();
      for (int64_t i = num_old - 1; i >= 0; i--) {
        al.at(i) = old_stps.at(i).dot(q) * ro.at(i);
        q.add_(old_dirs.at(i), torch::Scalar(-al.at(i)));
      }

      // Multiply by initial Hessian
      // r/d is the final direction
      Tensor r = q.mul(H_diag);
      d = r;

      for (int64_t i = 0; i < num_old; i++) {
        Tensor be_i = old_dirs.at(i).dot(r) * ro.at(i);
        r.add_(old_stps.at(i), torch::Scalar(al.at(i) - be_i));
      }
      prev_flat_grad.copy_(flat_grad);
    }

    /**
     * comute step length
     */

    // reset initial guess for step size
    if (n_iter == 1) {
      t = torch::Scalar(
          at::min(ONE, ONE / abs_grad_sum) * options.learning_rate_);
    } else {
      t = options.learning_rate_;
    }

    torch::Scalar gtd = torch::Scalar(flat_grad.dot(d));
    add_grad(t, d);
    int64_t ls_func_evals = 0;
    if (n_iter != options.max_iter_) {
      // re-evaluate function only if not in last iteration
      // the reason we do this: in a stochastic setting,
      // no use to re-evaluate that function here
      loss = closure();
      flat_grad = gather_flat_grad();
      abs_grad_sum = torch::Scalar(flat_grad.abs().sum());
      ls_func_evals = 1;
    }

    current_evals += ls_func_evals;

    /**
     * Check conditions
     */

    if (n_iter == options.max_iter_) {
      break;
    } else if (current_evals >= options.max_eval_) {
      break;
    } else if (abs_grad_sum.toFloat() <= options.tolerance_grad_) {
      break;
    } else if (gtd.toFloat() > -options.tolerance_grad_) {
      break;
    } else if (
        torch::Scalar(d.mul(t).abs_().sum()).toFloat() <=
        options.tolerance_change_) {
      break;
    } else if (
        std::abs(loss.toCFloat() - prev_loss.toFloat()) <
        options.tolerance_change_) {
      break;
    }
  }
  return orig_loss;
}
} // namespace optim
} // namespace torch
