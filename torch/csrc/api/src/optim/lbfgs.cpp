#include <torch/optim/lbfgs.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

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
    if (!parameter.grad().defined()) {
      views.push_back(parameter.new_empty({parameter.numel()}).zero_());
    }
    else if (parameter.grad().is_sparse()) {
      views.push_back(parameter.grad().to_dense().view(-1));
    }
    else {
      views.push_back(parameter.grad().view(-1));
    }
  }
  return torch::cat(views);
}

void LBFGS::add_grad(const torch::Tensor& step_size, const Tensor& update) {
  NoGradGuard guard;
  int64_t offset = 0;
  for (auto& parameter : parameters_) {
    int64_t numel = parameter.numel();
    parameter.add_(
        update.slice(0, offset, offset + numel, 1).view_as(parameter),
        step_size.item<float>());
    offset += numel;
  }
}

torch::Tensor LBFGS::step(LossClosure closure) {
  torch::Tensor orig_loss = closure();
  torch::Tensor loss = orig_loss.clone(at::MemoryFormat::Contiguous);
  int64_t current_evals = 1;
  func_evals += 1;

  Tensor flat_grad = gather_flat_grad();
  Tensor abs_grad_sum = flat_grad.abs().sum();

  if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
    return loss;
  }

  Tensor ONE = torch::tensor(1, flat_grad.options());

  int64_t n_iter = 0;
  while (n_iter < options.max_iter()) {
    n_iter++;
    state_n_iter++;

    if (state_n_iter == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
      prev_flat_grad = flat_grad.clone(at::MemoryFormat::Contiguous);
    } else {
      Tensor y = flat_grad.sub(prev_flat_grad);
      Tensor s = d.mul(t);
      Tensor ys = y.dot(s);

      if (ys.item<float>() > 1e-10) {
        // updating memory

        if (old_dirs.size() == options.history_size()) {
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
        q.add_(old_dirs.at(i), -al.at(i).item());
      }

      // Multiply by initial Hessian
      // r/d is the final direction
      Tensor r = q.mul(H_diag);
      d = r;

      for (int64_t i = 0; i < num_old; i++) {
        Tensor be_i = old_dirs.at(i).dot(r) * ro.at(i);
        r.add_(old_stps.at(i), (al.at(i) - be_i).item());
      }
      prev_flat_grad.copy_(flat_grad);
    }

    /**
     * compute step length
     */

    // reset initial guess for step size
    if (n_iter == 1) {
      t = torch::min(ONE, ONE / abs_grad_sum) * options.learning_rate();
    } else {
      t = torch::tensor(options.learning_rate(), flat_grad.options());
    }

    Tensor gtd = flat_grad.dot(d);
    add_grad(t, d);
    int64_t ls_func_evals = 0;
    if (n_iter != options.max_iter()) {
      // re-evaluate function only if not in last iteration
      // the reason we do this: in a stochastic setting,
      // no use to re-evaluate that function here
      loss = closure();
      flat_grad = gather_flat_grad();
      abs_grad_sum = flat_grad.abs().sum();
      ls_func_evals = 1;
    }

    current_evals += ls_func_evals;

    /**
     * Check conditions
     */

    if (n_iter == options.max_iter()) {
      break;
    } else if (current_evals >= options.max_eval()) {
      break;
    } else if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
      break;
    } else if (gtd.item<float>() > -options.tolerance_grad()) {
      break;
    } else if (
        d.mul(t).abs_().sum().item<float>() <= options.tolerance_change()) {
      break;
    } else if (
        std::abs(loss.item<float>() - prev_loss.item<float>()) <
        options.tolerance_change()) {
      break;
    }
  }
  return orig_loss;
}

void LBFGS::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
