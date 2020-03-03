#include <torch/optim/lbfgs.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>

namespace torch {
namespace optim {

LBFGSOptions::LBFGSOptions(double lr)
    : lr_(lr) {}

Tensor LBFGS::_gather_flat_grad() {
  std::vector<Tensor> views;
  for (auto& parameter : _params) {
    if (!parameter.grad().defined()) {
      views.emplace_back(parameter.new_empty({parameter.numel()}).zero_());
    }
    else if (parameter.grad().is_sparse()) {
      views.push_back(parameter.grad().to_dense().view(-1));
    }
    else {
      views.push_back(parameter.grad().view(-1));
    }
  }
  return torch::cat(views, 0);
}

int64_t LBFGS::_numel() {
  if(_numel_cache == c10::nullopt) {
    auto res = 0;
    for (auto& parameter : _params) {
      int64_t numel = parameter.numel();
      res+=numel;
    }
    *_numel_cache = res;
  }
  return *_numel_cache;
}

void LBFGS::add_grad(const torch::Tensor& step_size, const Tensor& update) {
  NoGradGuard guard;
  int64_t offset = 0;
  for (auto& parameter : _params) {
    int64_t numel = parameter.numel();
    parameter.add_(
        update.slice(0, offset, offset + numel, 1).view_as(parameter),
        step_size.item<double>());
    offset += numel;
  }
  TORCH_INTERNAL_ASSERT(offset == _numel())
}

void LBFGS::step() {
  TORCH_INTERNAL_ASSERT(param_groups_.size() == 1);
  auto& group = param_groups_[0];
  auto options = static_cast<LBFGSOptions&>(group.options());
  auto lr = options.lr();
  auto max_iter = options.max_iter();
  auto max_eval = options.max_eval();
  auto tolerance_grad = options.tolerance_grad();
  auto tolerance_change = options.tolerance_change();
  auto line_search_fn = options.line_search_fn();
  auto history_size = options.history_size();

  // NOTE: LBFGS has only global state, but we register it as state for
  // the first param, because this helps with casting in load_state_dict
  auto param_state = state_.find(c10::guts::to_string(_params[0].unsafeGetTensorImpl()));
  if(param_state == state_.end()) {
    state_[c10::guts::to_string(_params[0].unsafeGetTensorImpl())] = std::make_unique<LBFGSParamState>();
  }
  auto& state = static_cast<LBFGSParamState&>(*state_[c10::guts::to_string(_params[0].unsafeGetTensorImpl())]);

  // evaluate initial f(x) and df/dx
  auto orig_loss = torch::zeros(2); //TODO: closure()
  auto loss = orig_loss.clone(at::MemoryFormat::Contiguous);
  auto current_evals = 1;
  state.func_evals(state.func_evals()+1);

  auto flat_grad = _gather_flat_grad();
  auto opt_cond = (flat_grad.abs().max().item<double>() <= tolerance_grad);

  // optimal condition
  if(opt_cond) {
    // return orig_loss;
  }
  // tensors cached in state (for tracing)
  auto& d = state.d();
  auto& t = state.t();
  auto& old_dirs = state.old_dirs();
  auto& old_stps = state.old_stps();
  auto& ro = state.ro();
  auto& H_diag = state.H_diag();
  auto& prev_flat_grad = state.prev_flat_grad();
  auto& prev_loss = state.prev_loss();
  auto& al = state.al();
  int n_iter = 0;
  Tensor ONE = torch::tensor(1, flat_grad.options());

  // optimize for a max of max_iter iterations
  while(n_iter < max_iter) {
    // keep track of nb of iterations
    n_iter += 1;
    state.n_iter(state.n_iter()+1);

    // compute gradient descent direction
    if (state.n_iter() == 1) {
      d = flat_grad.neg();
      H_diag = ONE;
    } else {
      // do lbfgs update (update memory)
      auto y = flat_grad.sub(prev_flat_grad);
      auto s = d.mul(t);
      auto ys = y.dot(s); // y*s
      if (ys.item<double>() > 1e-10) {
        // updating memory
        if (old_dirs.size() == history_size) {
          // shift history by one (limited-memory)
          old_dirs.pop_front();
          old_stps.pop_front();
          ro.pop_front();
        }
        // store new direction/step
        old_dirs.emplace_back(y);
        old_stps.emplace_back(s);
        ro.emplace_back(1. / ys);

        // update scale of initial Hessian approximation
        H_diag = ys / y.dot(y);  // (y*y)
      }

      // compute the approximate (L-BFGS) inverse Hessian
      // multiplied by the gradient
      auto num_old = old_dirs.size();

      if(al.size() == 0) {
        al.resize(history_size);
      }

      // iteration in L-BFGS loop collapsed to use just one buffer
      auto q = flat_grad.neg();
      for (int i = num_old - 1; i > -1; i--) {
        al[i] = old_stps[i].dot(q) * ro[i];
        q.add_(old_dirs[i], -al[i].item<double>());
      }
    }

    if (!prev_flat_grad.defined()) {
      prev_flat_grad = flat_grad.clone(at::MemoryFormat::Contiguous);
    } else {
      prev_flat_grad.copy_(flat_grad);
    }
    prev_loss = loss;

    // ############################################################
    // # compute step length
    // ############################################################
    // # reset initial guess for step size
    if (state.n_iter() == 1) {
      t = std::min(1., 1. / flat_grad.abs().sum().item<double>());
    } else {
      t = lr;
    }

    // directional derivative
    auto gtd = flat_grad.dot(d);  // g * d

    // directional derivative is below tolerance
    // if (torch::gt(gtd > -torch::Tensor(tolerance_change))) break;

    // optional line search: user function
    auto ls_func_evals = 0;
    if (line_search_fn != c10::nullopt) {
      //TORCH_CHECK(*line_search_fn == strong_wolfe, "only 'strong_wolfe' is supported");
      if(true) {

      } else {
        // x_init = self._clone_param()
        // define obj_func
        // call string wolfe
      }
    } else {
      // # no line search, simply move with fixed-step
      // self._add_grad(t, d)
      // if n_iter != max_iter:
      //     # re-evaluate function only if not in last iteration
      //     # the reason we do this: in a stochastic setting,
      //     # no use to re-evaluate that function here
      //     loss = float(closure())
      //     flat_grad = self._gather_flat_grad()
      //     opt_cond = flat_grad.abs().max() <= tolerance_grad
      //     ls_func_evals = 1
    }

    // update func eval
    current_evals += ls_func_evals;
    state.func_evals(state.func_evals() + ls_func_evals);

    // ############################################################
    // # check conditions
    // ############################################################
    if (n_iter == max_iter) break;
    if (max_eval != c10::nullopt && current_evals >= *max_eval) break;
    // optimal condition
    if (opt_cond) break;
    // lack of progress
    if (d.mul(t).abs().max().item<float>() <= tolerance_change) break;
    if (abs(loss.item<float>() - prev_loss.item<float>()) < tolerance_change) break;

    //return orig_loss
  }
  // state['d'] = d
  // state['t'] = t
  // state['old_dirs'] = old_dirs
  // state['old_stps'] = old_stps
  // state['ro'] = ro
  // state['H_diag'] = H_diag
  // state['prev_flat_grad'] = prev_flat_grad
  // state['prev_loss'] = prev_loss
}
//
// torch::Tensor LBFGS::step(LossClosure closure) {
//   torch::Tensor orig_loss = closure();
//   // torch::Tensor loss = orig_loss.clone(at::MemoryFormat::Contiguous);
//   // int64_t current_evals = 1;
//   // func_evals += 1;
//   //
//   // Tensor flat_grad = gather_flat_grad();
//   // Tensor abs_grad_sum = flat_grad.abs().sum();
//   //
//   // if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
//   //   return loss;
//   // }
//   //
//   // Tensor ONE = torch::tensor(1, flat_grad.options());
//   //
//   // int64_t n_iter = 0;
//   // while (n_iter < options.max_iter()) {
//   //   n_iter++;
//   //   state_n_iter++;
//   //
//   //   if (state_n_iter == 1) {
//   //     d = flat_grad.neg();
//   //     H_diag = ONE;
//   //     prev_flat_grad = flat_grad.clone(at::MemoryFormat::Contiguous);
//   //   } else {
//   //     Tensor y = flat_grad.sub(prev_flat_grad);
//   //     Tensor s = d.mul(t);
//   //     Tensor ys = y.dot(s);
//   //
//   //     if (ys.item<float>() > 1e-10) {
//   //       // updating memory
//   //
//   //       if (old_dirs.size() == options.history_size()) {
//   //         // shift history by one (limited memory)
//   //         old_dirs.pop_front();
//   //         old_stps.pop_front();
//   //       }
//   //
//   //       // store new direction/step
//   //       old_dirs.push_back(y);
//   //       old_stps.push_back(s);
//   //
//   //       // update scale of initial Hessian approximation
//   //       H_diag = ys / y.dot(y);
//   //     }
//   //
//   //     int64_t num_old = old_dirs.size();
//   //
//   //     for (int64_t i = 0; i < num_old; i++) {
//   //       ro.at(i) = ONE / old_dirs.at(i).dot(old_stps.at(i));
//   //     }
//   //
//   //     Tensor q = flat_grad.neg();
//   //     for (int64_t i = num_old - 1; i >= 0; i--) {
//   //       al.at(i) = old_stps.at(i).dot(q) * ro.at(i);
//   //       q.add_(old_dirs.at(i), -al.at(i).item());
//   //     }
//   //
//   //     // Multiply by initial Hessian
//   //     // r/d is the final direction
//   //     Tensor r = q.mul(H_diag);
//   //     d = r;
//   //
//   //     for (int64_t i = 0; i < num_old; i++) {
//   //       Tensor be_i = old_dirs.at(i).dot(r) * ro.at(i);
//   //       r.add_(old_stps.at(i), (al.at(i) - be_i).item());
//   //     }
//   //     prev_flat_grad.copy_(flat_grad);
//   //   }
//   //
//   //   /**
//   //    * compute step length
//   //    */
//   //
//   //   // reset initial guess for step size
//   //   if (n_iter == 1) {
//   //     t = torch::min(ONE, ONE / abs_grad_sum) * options.learning_rate();
//   //   } else {
//   //     t = torch::tensor(options.learning_rate(), flat_grad.options());
//   //   }
//   //
//   //   Tensor gtd = flat_grad.dot(d);
//   //   add_grad(t, d);
//   //   int64_t ls_func_evals = 0;
//   //   if (n_iter != options.max_iter()) {
//   //     // re-evaluate function only if not in last iteration
//   //     // the reason we do this: in a stochastic setting,
//   //     // no use to re-evaluate that function here
//   //     loss = closure();
//   //     flat_grad = gather_flat_grad();
//   //     abs_grad_sum = flat_grad.abs().sum();
//   //     ls_func_evals = 1;
//   //   }
//   //
//   //   current_evals += ls_func_evals;
//   //
//   //   /**
//   //    * Check conditions
//   //    */
//   //
//   //   if (n_iter == options.max_iter()) {
//   //     break;
//   //   } else if (current_evals >= options.max_eval()) {
//   //     break;
//   //   } else if (abs_grad_sum.item<float>() <= options.tolerance_grad()) {
//   //     break;
//   //   } else if (gtd.item<float>() > -options.tolerance_grad()) {
//   //     break;
//   //   } else if (
//   //       d.mul(t).abs_().sum().item<float>() <= options.tolerance_change()) {
//   //     break;
//   //   } else if (
//   //       std::abs(loss.item<float>() - prev_loss.item<float>()) <
//   //       options.tolerance_change()) {
//   //     break;
//   //   }
//   // }
//   return orig_loss;
// }
//
void LBFGS::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
