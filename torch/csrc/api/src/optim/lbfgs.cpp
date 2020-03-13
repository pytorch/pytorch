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

bool operator==(const LBFGSOptions& lhs, const LBFGSOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
         (lhs.max_iter() == rhs.max_iter()) &&
         (lhs.max_eval() == rhs.max_eval()) &&
         (lhs.tolerance_grad() == rhs.tolerance_grad()) &&
         (lhs.tolerance_change() == rhs.tolerance_change() &&
         (lhs.history_size() == rhs.history_size()));
}

void LBFGSOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_iter);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_eval);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_grad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(tolerance_change);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(history_size);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(line_search_fn);
}

void LBFGSOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, max_iter);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(int64_t, max_eval);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_change);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, history_size);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(std::string, line_search_fn);
}

template <typename T>
bool if_container_equal(T lhs, T rhs) {
  if(lhs.size() == rhs.size()) return false;
  for (int i = 0; i < lhs.size(); i++) {
    if(!torch::equal(lhs[i], rhs[i])) return false;
  }
  return true;
}

bool operator==(const LBFGSParamState& lhs, const LBFGSParamState& rhs) {
  return (lhs.func_evals() == rhs.func_evals()) &&
         (lhs.n_iter() == rhs.n_iter()) &&
         (lhs.t() == rhs.t()) &&
         torch::equal_if_defined(lhs.d(), lhs.d()) &&
         torch::equal_if_defined(lhs.H_diag(), rhs.H_diag()) &&
         torch::equal_if_defined(lhs.prev_flat_grad(), rhs.prev_flat_grad()) &&
         torch::equal_if_defined(lhs.prev_loss(), rhs.prev_loss()) &&
         if_container_equal(lhs.old_dirs(), rhs.old_dirs()) &&
         if_container_equal(lhs.old_stps(), rhs.old_stps()) &&
         if_container_equal(lhs.ro(), rhs.ro()) &&
         if_container_equal(lhs.al(), rhs.al());
}

void LBFGSParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(func_evals);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(n_iter);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(t);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(d);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(H_diag);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_flat_grad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_loss);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(old_dirs);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(old_stps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(ro);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(al);
}

void LBFGSParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, func_evals);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, n_iter);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, t);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, d);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, H_diag);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev_flat_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev_loss);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, old_dirs);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, old_stps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(std::deque<Tensor>, ro);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(std::vector<Tensor>, al);
}

Tensor LBFGS::_gather_flat_grad() {
  std::vector<Tensor> views;
  for (const auto& p : *_params) {
    Tensor view;
    if (!p.grad().defined()) {
      views.emplace_back(p.new_empty({p.numel()}).zero_());
    }
    else if (p.grad().is_sparse()) {
      views.emplace_back(p.grad().to_dense().view(-1));
    }
    else {
      views.emplace_back(p.grad().view(-1));
    }
  }
  return torch::cat(views, 0);
}

int64_t LBFGS::_numel() {
  if(_numel_cache == c10::nullopt) {
    auto res = 0;
    for (const auto& p :*_params) {
      res += p.numel();
    }
    _numel_cache = res;
  }
  return *_numel_cache;
}

void LBFGS::_add_grad(const double step_size, const Tensor& update) {
  auto offset = 0;
  for (auto& p : *_params) {
    auto numel = p.numel();
    // view as to avoid deprecated pointwise semantics
    p.add_(update.index(
      {at::indexing::Slice(offset, offset + numel)}).view_as(p), step_size);
    offset += numel;
  }
  TORCH_INTERNAL_ASSERT(offset == _numel());
}

void LBFGS::_set_param(const std::vector<Tensor>& params_data) {
  TORCH_INTERNAL_ASSERT(params_data.size() == _params->size());
  for (size_t i = 0; i < _params->size(); i++) {
    _params->at(i).copy_(params_data.at(i));
  }
}

std::vector<Tensor> LBFGS::_clone_param() {
  std::vector<Tensor> result;
  for (const auto& p : *_params) {
    result.emplace_back(p.clone(at::MemoryFormat::Contiguous));
  }
  return result;
}

std::tuple<Tensor, Tensor> LBFGS::_directional_evaluate(
  LossClosure closure, const std::vector<Tensor>& x, double t, const Tensor& d) {
    _add_grad(t, d);
    Tensor loss;
    {
      torch::AutoGradMode enable_grad(true);
      loss = closure();
    }
    auto flat_grad = _gather_flat_grad();
    _set_param(x);
    return std::make_tuple(loss, flat_grad);
}

double _cubic_interpolate(
  double x1, double f1, double g1, double x2, double f2, double g2,
  c10::optional<std::tuple<double, double>> bounds = c10::nullopt) {
  // ported from https://github.com/torch/optim/blob/master/polyinterp.lua
  // Compute bounds of interpolation area
  double xmin_bound, xmax_bound;
  if (bounds != c10::nullopt) {
    xmin_bound = std::get<0>(*bounds);
    xmin_bound = std::get<1>(*bounds);
  } else {
    xmin_bound = (x1 <= x2) ? x1 : x2;
    xmax_bound = (x1 <= x2) ? x2 : x1;
  }
  // Code for most common case: cubic interpolation of 2 points
  //   w/ function and derivative values for both
  // Solution in this case (where x2 is the farthest point):
  //   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
  //   d2 = sqrt(d1^2 - g1*g2);
  //   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
  //   t_new = min(max(min_pos,xmin_bound),xmax_bound);

  auto d1 = (g1 + g2) - (3 * (f1 - f2) / (x1 - x2));
  auto d2_square = std::pow(d1, 2) - g1 * g2;
  double d2;
  if (d2_square >= 0) {
    d2 = std::sqrt(d2_square);
    double min_pos;
    if(x1 <= x2) {
      min_pos = x2 - ((x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)));
    } else {
      min_pos = x1 - ((x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)));
    }
    return std::min(std::max(min_pos, xmin_bound), xmax_bound);
  } else {
    return (xmin_bound + xmax_bound) / 2;
  }
}

template <typename Function>
std::tuple<Tensor, Tensor, double, int64_t> _strong_wolfe(Function obj_func, const std::vector<Tensor>& x,
                                          double t, Tensor d, Tensor f, Tensor g, Tensor gtd, double c1 = 1e-4,
                                          double c2 = 0.9, double tolerance_change = 1e-9, double max_ls = 25) {

    auto val = [](const Tensor& t) {return t.item<double>(); };

    auto d_norm = val(d.abs().max());
    g = g.clone(at::MemoryFormat::Contiguous);
    // evaluate objective and gradient using initial step
    auto obj_func_res = obj_func(x, t, d);
    auto f_new = std::get<0>(obj_func_res);
    auto g_new = std::get<1>(obj_func_res);
    auto ls_func_evals = 1;
    auto gtd_new = g_new.dot(d);

    // bracket an interval containing a point satisfying the Wolfe criteria
    auto t_prev = 0;
    auto f_prev = f;
    auto g_prev = g;
    auto gtd_prev = gtd;
    bool done = false;
    auto ls_iter = 0;
    std::vector<double> bracket;
    std::vector<Tensor> bracket_f, bracket_g, bracket_gtd;

    while (ls_iter < max_ls) {
      // check conditions
      if ((val(f_new) > val(f + c1 * t * gtd)) || (ls_iter > 1 && (val(f_new) >= val(f_prev)))) {
        bracket = {t_prev, t};
        bracket_f = {f_prev, f_new};
        bracket_g = {g_prev, g_new.clone(at::MemoryFormat::Contiguous)};
        bracket_gtd = {gtd_prev, gtd_new};
        break;
      }
      if (abs(val(gtd_new)) <= (-c2 * val(gtd))) {
        bracket = {t, t};
        bracket_f = {f_new, f_new};
        bracket_g = {g_new, g_new};
        done = true;
        break;
      }
      if (val(gtd_new) >= 0) {
        bracket = {t_prev, t};
        bracket_f = {f_prev, f_new};
        bracket_g = {g_prev, g_new.clone(at::MemoryFormat::Contiguous)};
        bracket_gtd = {gtd_prev, gtd_new};
        break;
      }
      // interpolate
      auto min_step = t + 0.01 * (t - t_prev);
      auto max_step = t * 10;
      auto tmp = t;
      t = _cubic_interpolate(t_prev, val(f_prev), val(gtd_prev),
                             t, val(f_new), val(gtd_new),
                             std::make_tuple(min_step, max_step));
      // next step
      t_prev = tmp;
      f_prev = f_new;
      g_prev = g_new.clone(at::MemoryFormat::Contiguous);
      gtd_prev = gtd_new;
      obj_func_res = obj_func(x, t, d);
      f_new = std::get<0>(obj_func_res);
      g_new = std::get<1>(obj_func_res);
      ls_func_evals += 1;
      gtd_new = g_new.dot(d);
      ls_iter += 1;
    }
    // reached max number of iterations?
    if (ls_iter == max_ls) {
      bracket = {0, t};
      bracket_f = {f, f_new};
      bracket_g = {g, g_new};
    }

    // zoom phase: we now have a point satisfying the criteria, or
    // a bracket around it. We refine the bracket until we find the
    // exact point satisfying the criteria
    bool insuf_progress = false;
    // find high and low points in bracket
    int64_t low_pos, high_pos;
    if(val(bracket_f[0]) <= val(bracket_f[1])) {
      low_pos = 0;
      high_pos = 1;
    } else {
      low_pos = 1;
      high_pos = 0;
    }
    while(!done && ls_iter < max_ls) {
      // compute new trial value
      t = _cubic_interpolate(bracket[0], val(bracket_f[0]), val(bracket_gtd[0]),
                             bracket[1], val(bracket_f[1]), val(bracket_gtd[1]));

      // test that we are making sufficient progress:
      // in case `t` is so close to boundary, we mark that we are making
      // insufficient progress, and if
      //   + we have made insufficient progress in the last step, or
      //   + `t` is at one of the boundary,
      // we will move `t` to a position which is `0.1 * len(bracket)`
      // away from the nearest boundary point.
      double bracket_max = std::max(bracket[0], bracket[1]);
      auto bracket_min = std::min(bracket[0], bracket[1]);
      auto eps = 0.1 * (bracket_max - bracket_min);
      if (std::min(bracket_max - t, t - bracket_min) < eps) {
        // interpolation close to boundary
        if (insuf_progress || (t >= bracket_max) || (t <= bracket_min)) {
          // evaluate at 0.1 away from boundary
          t = (abs(t - bracket_max) < abs(t - bracket_min)) ? bracket_max - eps : bracket_min + eps;
          insuf_progress = false;
        } else {
          insuf_progress = true;
        }
      } else {
        insuf_progress = false;
      }

      // Evaluate new point
      obj_func_res = obj_func(x, t, d);
      f_new = std::get<0>(obj_func_res);
      g_new = std::get<1>(obj_func_res);
      ls_func_evals += 1;
      gtd_new = g_new.dot(d);
      ls_iter += 1;

      if ((val(f_new) > val(f + c1 * t * gtd)) || (val(f_new) >= val(bracket_f[low_pos]))) {
        // Armijo condition not satisfied or not lower than lowest point
        // # Armijo condition not satisfied or not lower than lowest point
        bracket[high_pos] = t;
        bracket_f[high_pos] = f_new;
        bracket_g[high_pos] = g_new.clone(at::MemoryFormat::Contiguous);
        bracket_gtd[high_pos] = gtd_new;
        low_pos = (val(bracket_f[0]) <= val(bracket_f[1])) ? 0 : 1;
        high_pos = (val(bracket_f[0]) <= val(bracket_f[1])) ? 1 : 0;
      } else {
        if (val(abs(gtd_new)) <= (-c2 * val(gtd))) {
          // Wolfe conditions satisfied
          done = true;
        } else if (val(gtd_new * (bracket[high_pos] - bracket[low_pos])) >= 0) {
          // old high becomes new low
          bracket[high_pos] = bracket[low_pos];
          bracket_f[high_pos] = bracket_f[low_pos];
          bracket_g[high_pos] = bracket_g[low_pos];
          bracket_gtd[high_pos] = bracket_gtd[low_pos];
        }

        // new point becomes new low
        bracket[low_pos] = t;
        bracket_f[low_pos] = f_new;
        bracket_g[low_pos] = g_new.clone(at::MemoryFormat::Contiguous);
        bracket_gtd[low_pos] = gtd_new;
      }

      // line-search bracket is so small
      if((abs(bracket[1] - bracket[0]) * d_norm) < tolerance_change) break;
    }

    // return stuff
    t = bracket[low_pos];
    f_new = bracket_f[low_pos];
    g_new = bracket_g[low_pos];
    return std::make_tuple(f_new, g_new, t, ls_func_evals);
}

Tensor LBFGS::step(LossClosure closure) {
  NoGradGuard no_grad;
  TORCH_CHECK(closure != nullptr, "LBFGS requires a closure function");
  TORCH_INTERNAL_ASSERT(param_groups_.size() == 1);
  auto val = [](const Tensor& t) {return t.item<double>(); };

  auto& group = param_groups_[0];
  const auto& options = static_cast<const LBFGSOptions&>(group.options());
  auto lr = options.lr();
  auto max_iter = options.max_iter();
  auto max_eval = options.max_eval();
  auto tolerance_grad = options.tolerance_grad();
  auto tolerance_change = options.tolerance_change();
  auto line_search_fn = options.line_search_fn();
  auto history_size = options.history_size();

  // NOTE: LBFGS has only global state, but we register it as state for
  // the first param, because this helps with casting in load_state_dict
  auto param_state = state_.find(c10::guts::to_string(_params->at(0).unsafeGetTensorImpl()));
  if(param_state == state_.end()) {
    state_[c10::guts::to_string(_params->at(0).unsafeGetTensorImpl())] = std::make_unique<LBFGSParamState>();
  }
  auto& state = static_cast<LBFGSParamState&>(*state_[c10::guts::to_string(_params->at(0).unsafeGetTensorImpl())]);
  // evaluate initial f(x) and df/dx
  Tensor orig_loss;
  {
    torch::AutoGradMode enable_grad(true);
    orig_loss = closure();
  }

  auto loss = orig_loss.clone(at::MemoryFormat::Contiguous);
  auto current_evals = 1;
  state.func_evals(state.func_evals()+1);
  auto flat_grad = _gather_flat_grad();
  auto opt_cond = (val(flat_grad.abs().max()) <= tolerance_grad);

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
      if (val(ys) > 1e-10) {
        // updating memory
        if (int(old_dirs.size()) == history_size) {
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
      for (int i = num_old - 1; i >= 0; i--) {
        al.at(i) = old_stps.at(i).dot(q) * ro.at(i);
        q.add_(old_dirs.at(i), -val(al[i]));
      }

      // multiply by initial Hessian
      // r/d is the final direction
      d = torch::mul(q, H_diag);
      for (size_t i = 0; i < num_old; i++) {
        auto be_i = old_dirs[i].dot(d) * ro[i];
        d.add_(old_stps[i], (al[i] - be_i).item<double>());
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
      t = std::min(1., 1. / val(flat_grad.abs().sum())) * lr;
    } else {
      t = lr;
    }

    // directional derivative
    auto gtd = flat_grad.dot(d);  // g * d

    // directional derivative is below tolerance
    if (val(gtd) > -tolerance_change) break;

    // optional line search: user function
    auto ls_func_evals = 0;
    if (line_search_fn != c10::nullopt) {
      TORCH_CHECK(*line_search_fn == "strong_wolfe", "only 'strong_wolfe' is supported");
      auto x_init = _clone_param();
      auto obj_func = [&](const std::vector<Tensor>& x, double t, const Tensor& d) { return _directional_evaluate(closure, x, t, d); };
      auto strong_wolfe_res = _strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd);
      _add_grad(t, d);
    } else {
      // no line search, simply move with fixed-step
      _add_grad(t, d);
      if (n_iter != max_iter) {
        // re-evaluate function only if not in last iteration
        // the reason we do this: in a stochastic setting,
        // no use to re-evaluate that function here
        {
          torch::AutoGradMode enable_grad(true);
          loss = closure();
        }
        flat_grad = _gather_flat_grad();
        opt_cond = val(torch::max(flat_grad.abs())) <= tolerance_grad;
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

    if (current_evals >= *max_eval) break;

    // optimal condition
    if (opt_cond) break;

    // lack of progress
    if (val(d.mul(t).abs().max()) <= tolerance_change) break;

    if (val(abs(loss - prev_loss)) < tolerance_change) break;
  }

  return orig_loss;
}

void LBFGS::add_parameters(const std::vector<Tensor>& parameters) {
  return _add_parameters_new_design(parameters);
}

const std::vector<Tensor>& LBFGS::parameters() const noexcept {
  return _parameters_new_design();
}

std::vector<Tensor>& LBFGS::parameters() noexcept {
  return _parameters_new_design();
}

size_t LBFGS::size() const noexcept {
  return _size_new_design();
}

void LBFGS::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void LBFGS::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  }
  else { // deserializing archives saved in old format (prior to version 1.5.0)
    TORCH_WARN(
      "Your serialized Adagrad optimizer is still using the old serialization format. "
      "The func_evals and n_iter value in state will be set to 0 because the old RMSprop optimizer didn't save these values."
      "You should re-save your Adagrad optimizer to use the new serialization format.");
    Tensor d, t, H_diag, prev_flat_grad, prev_loss;
    std::deque<Tensor> old_dirs, old_stps;
    archive("d", d);
    archive("t", t);
    archive("H_diag", H_diag);
    archive("prev_flat_grad", prev_flat_grad);
    archive("prev_loss", prev_loss);
    torch::optim::serialize(archive, "old_dirs", old_dirs);
    torch::optim::serialize(archive, "old_stps", old_stps);
    std::vector<Tensor> params = *_params;
    for (size_t idx = 0; idx < params.size(); idx++) {
      auto state = std::make_unique<LBFGSParamState>();
      state->d(d);
      state->t(t.item<double>());
      state->H_diag(H_diag);
      state->prev_flat_grad(prev_flat_grad);
      state->prev_loss(prev_loss);
      state->old_dirs(old_dirs);
      state->old_stps(old_stps);
      state_[c10::guts::to_string(params[idx].unsafeGetTensorImpl())] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
