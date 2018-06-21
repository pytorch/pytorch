#include <torch/optim/lbfgs.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>
#include <vector>

namespace torch {
namespace optim {
at::Tensor LBFGS::gather_flat_grad() {
  std::vector<at::Tensor> views;
  for (auto& parameter : model_->parameters()) {
    views.push_back(
        torch::autograd::as_variable_ref(parameter->grad()).data().view(-1));
  }
  return at::cat(views);
}

void LBFGS::add_grad(const at::Scalar& step_size, const at::Tensor& update) {
  int offset = 0;
  for (auto& parameter : model_->parameters()) {
    int numel = parameter->numel();
    at::Tensor& pd = parameter->data();
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

  if (at::Scalar(abs_grad_sum).toFloat() <= tolerance_grad_) {
    return loss;
  }

  at::Tensor ONE = flat_grad.type().scalarTensor(1);

  int n_iter = 0;
  while (n_iter < max_iter_) {
    n_iter++;
    state_n_iter++;

    if (state_n_iter == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
      prev_flat_grad = flat_grad.clone();
    } else {
      at::Tensor y = flat_grad.sub(prev_flat_grad);
      at::Tensor s = d.mul(t);
      at::Scalar ys = at::Scalar(y.dot(s));

      if (ys.toFloat() > 1e-10) {
        // updating memory

        if ((int)old_dirs.size() == history_size_) {
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

      for (int i = 0; i < num_old; i++) {
        ro[i] = ONE / old_dirs[i].dot(old_stps[i]);
      }

      at::Tensor q = flat_grad.neg();
      for (int i = num_old - 1; i >= 0; i--) {
        al[i] = old_stps[i].dot(q) * ro[i];
        q.add_(old_dirs[i], at::Scalar(-al[i]));
      }

      // Multiply by initial Hessian
      // r/d is the final direction
      at::Tensor r = q.mul(H_diag);
      d = r;

      for (int i = 0; i < num_old; i++) {
        at::Tensor be_i = old_dirs[i].dot(r) * ro[i];
        r.add_(old_stps[i], at::Scalar(al[i] - be_i));
      }
      prev_flat_grad.copy_(flat_grad);
    }

    /**
     * comute step length
     */

    // reset initial guess for step size
    if (n_iter == 1) {
      t = at::Scalar(at::min(ONE, ONE / abs_grad_sum) * lr_);
    } else {
      t = lr_;
    }

    at::Scalar gtd = at::Scalar(flat_grad.dot(d));
    add_grad(t, d);
    int ls_func_evals = 0;
    if (n_iter != max_iter_) {
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

    if (n_iter == max_iter_) {
      break;
    } else if (current_evals >= max_eval_) {
      break;
    } else if (abs_grad_sum.toFloat() <= tolerance_grad_) {
      break;
    } else if (gtd.toFloat() > -tolerance_grad_) {
      break;
    } else if (
        at::Scalar(d.mul(t).abs_().sum()).toFloat() <= tolerance_change_) {
      break;
    } else if (
        std::abs(loss.toFloat() - prev_loss.toFloat()) < tolerance_change_) {
      break;
    }
  }
  return orig_loss;
}

void LBFGS::init_state() {
  d = torch::empty({0});
  t = 0;
  H_diag = torch::empty({0});
  prev_flat_grad = torch::empty({0});
  prev_loss = 0;
  ro.resize(history_size_);
  al.resize(history_size_);
  func_evals = 0;
  state_n_iter = 0;
}

} // namespace optim
} // namespace torch
