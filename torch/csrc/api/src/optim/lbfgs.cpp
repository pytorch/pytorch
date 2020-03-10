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

LBFGSOptions::LBFGSOptions(double lr) : lr_(lr) {}

// TORCH_ARG(double, lr) = 1;
// TORCH_ARG(int64_t, max_iter) = 20;
// TORCH_ARG(c10::optional<int64_t>, max_eval) = c10::nullopt;
// TORCH_ARG(double, tolerance_grad) = 1e-7;
// TORCH_ARG(double, tolerance_change) = 1e-9;
// TORCH_ARG(size_t, history_size) = 100;
// TORCH_ARG(c10::optional<std::string>, line_search_fn) = c10::nullopt;
bool operator==(const LBFGSOptions& lhs, const LBFGSOptions& rhs) {
  auto isNull = [](c10::optional<int64_t>& max_eval){ return max_eval == c10::nullopt; };
  return (lhs.lr() == rhs.lr()) &&
         (lhs.max_iter() == rhs.max_iter()) &&
         // ((isNull(lhs.max_eval()) && isNull(rhs.max_eval())) ||
         //   (!isNull(lhs.max_eval()) && !isNull(rhs.max_eval()) && (*lhs.max_eval() == *rhs.max_eval()))) &&
         (lhs.tolerance_grad() == rhs.tolerance_grad()) &&
         (lhs.tolerance_change() == rhs.tolerance_change() &&
         (lhs.history_size() == rhs.history_size()));
}

void LBFGSOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_iter);
  //_TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_eval);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_grad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_change);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(history_size);
  //_TORCH_OPTIM_SERIALIZE_TORCH_ARG(line_search_fn);
}

void LBFGSOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, max_iter);
  //_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(c10::optional<int64_t>, max_eval);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_change);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, history_size);
  //_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(c10::optional<int64_t>, line_search_fn);
}

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

void LBFGS::_add_grad(const double step_size, const Tensor& update) {
  NoGradGuard guard;
  int64_t offset = 0;
  for (auto& parameter : _params) {
    int64_t numel = parameter.numel();
    parameter.add_(
      update.slice(0, offset, offset + numel, 1).view_as(parameter), step_size);
    offset += numel;
  }
  TORCH_INTERNAL_ASSERT(offset == _numel())
}

Tensor LBFGS::step(LossClosure closure) {
  NoGradGuard guard;
  TORCH_CHECK(closure != nullptr, "LBFGS requires a closure function");
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
  auto orig_loss = closure();
  auto loss = orig_loss.clone(at::MemoryFormat::Contiguous);
  auto current_evals = 1;
  state.func_evals(state.func_evals()+1);

  auto flat_grad = _gather_flat_grad();
  auto opt_cond = (flat_grad.abs().max().item<double>() <= tolerance_grad);

  // optimal condition
  if(opt_cond) {
    return orig_loss;
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
    // reset initial guess for step size
    if (state.n_iter() == 1) {
      t = std::min(1., 1. / flat_grad.abs().sum().item<double>());
    } else {
      t = lr;
    }

    // directional derivative
    auto gtd = flat_grad.dot(d);  // g * d

    // directional derivative is below tolerance
    if (gtd.item<double>() > -tolerance_change) break;

    // optional line search: user function
    auto ls_func_evals = 0;
    if (line_search_fn != c10::nullopt) {
      TORCH_CHECK(*line_search_fn == "strong_wolfe", "only 'strong_wolfe' is supported");
      // x_init = self._clone_param()
      // define obj_func
      // call string wolfe
    } else {
      // no line search, simply move with fixed-step
      _add_grad(t, d);
      if (n_iter != max_iter) {
        // re-evaluate function only if not in last iteration
        // the reason we do this: in a stochastic setting,
        // no use to re-evaluate that function here
        loss = closure();
        flat_grad = _gather_flat_grad();
        opt_cond = torch::max(flat_grad.abs()).item<double>() <= tolerance_grad;
        ls_func_evals = 1;
      }
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

    return orig_loss;
  }
}

void LBFGS::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
