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
  auto isNull = [](c10::optional<int64_t> max_eval){ return max_eval == c10::nullopt; };
  return (lhs.lr() == rhs.lr()) &&
         (lhs.max_iter() == rhs.max_iter()) &&
         ((isNull(lhs.max_eval()) && isNull(rhs.max_eval())) ||
           (!isNull(lhs.max_eval()) && !isNull(rhs.max_eval()) && (*lhs.max_eval() == *rhs.max_eval()))) &&
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
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(c10::optional<int64_t>, max_eval);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, tolerance_change);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, history_size);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(c10::optional<std::string>, line_search_fn);
}

bool equal_if_defined(Tensor t1, Tensor t2) {
  return ((!t1.defined() && !t2.defined()) || (t1.defined() && t2.defined() && torch::equal(t1, t2)));
}

template <typename T>
bool if_container_equal(T lhs, T rhs) {
  TORCH_INTERNAL_ASSERT(lhs.size() == rhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    torch::equal(lhs[i], rhs[i]);
  }
}

bool operator==(const LBFGSParamState& lhs, const LBFGSParamState& rhs) {
  return equal_if_defined(lhs.d(), lhs.d()) &&
         (lhs.t() == rhs.t()) &&
         if_container_equal<std::deque<Tensor>>(lhs.old_dirs(), rhs.old_dirs()) &&
         if_container_equal<std::deque<Tensor>>(lhs.old_stps(), rhs.old_stps()) &&
         if_container_equal<std::deque<Tensor>>(lhs.ro(), rhs.ro()) &&
         equal_if_defined(lhs.H_diag(), rhs.H_diag()) &&
         equal_if_defined(lhs.prev_flat_grad(), rhs.prev_flat_grad()) &&
         equal_if_defined(lhs.prev_loss(), rhs.prev_loss()) &&
         if_container_equal<std::vector<Tensor>>(lhs.al(), rhs.al()) &&
         (lhs.func_evals() == rhs.func_evals()) &&
         (lhs.n_iter() == rhs.n_iter());
}

void LBFGSParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(d);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(t);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(old_dirs);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(old_stps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(ro);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(H_diag);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_flat_grad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(prev_loss);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(al);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(func_evals);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(n_iter);
}

void LBFGSParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, d);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, t);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(std::deque<Tensor>, old_dirs);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(std::deque<Tensor>, old_stps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(std::deque<Tensor>, ro);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, H_diag);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev_flat_grad);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, prev_loss);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(std::vector<Tensor>, al);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, func_evals);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, n_iter);
}

Tensor LBFGS::_gather_flat_grad() {
  std::vector<Tensor> views;
  for (const auto& parameter : _params) {
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
  int64_t offset = 0;
  for (auto& parameter : _params) {
    int64_t numel = parameter.numel();
    // view as to avoid deprecated pointwise semantics
    parameter.add_(update.index(
      {at::indexing::Slice(offset, offset + numel)}).view_as(parameter), step_size);
    offset += numel;
  }
  TORCH_INTERNAL_ASSERT(offset == _numel());
}

void LBFGS::_set_param(const Tensor& params_data) {
  for (size_t i = 0; i < _params.size(); i++) {
    _params[i].copy_(params_data);
  }
}

std::tuple<Tensor, Tensor> LBFGS::_directional_evaluate(
  LossClosure closure, const Tensor& x, double t, const Tensor& d) {
    _add_grad(t, d);
    auto loss = closure();
    auto flat_grad = _gather_flat_grad();
    _set_param(x);
    return std::make_tuple(loss, flat_grad);
}

double _cubic_interpolate(
  double x1, double f1, double g1, double x2, double f2, double g2, c10::optional<std::tuple<double, double>> bounds = c10::nullopt) {
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
  auto d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2);
  auto d2_square = std::pow(d1, 2) - g1 * g2;
  double d2;
  if (d2_square >= 0) {
    d2 = std::sqrt(d2_square);
    double min_pos;
    if(x1 >= x2) {
      min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2));
    } else {
      min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2));
    }
    return std::min(std::max(min_pos, xmin_bound), xmax_bound);
  } else {
    return (xmin_bound + xmax_bound) / 2;
  }
}

template <typename Function>
void _strong_wolfe(Function obj_func,
      Tensor x, double t, Tensor d, double f, Tensor g, double gtd, double c1=1e-4,
      double c2=0.9, double tolerance_change=1e-9, double max_ls=25) {
    auto d_norm = d.abs().max();
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
    std::vector<double> bracket, bracket_f, bracket_gtd;
    std::vector<Tensor> bracket_g;
    while (ls_iter < max_ls) {
      // check conditions
      if (f_new > (f + c1 * t * gtd) || (ls_iter > 1 and f_new >= f_prev)) {
        bracket = {t_prev, t};
        bracket_f = {f_prev, f_new};
        bracket_g = {g_prev, g_new};
        bracket_gtd = {gtd_prev, gtd_new};
        break;
      }
      if (abs(gtd_new) <= -c2 * gtd) {
        bracket = {t};
        bracket_f = {f_new};
        bracket_g = {g_new};
        done = true;
        break;
      }
      if (gtd_new >= 0) {
        bracket = {t_prev, t};
        bracket_f = {f_prev, f_new};
        bracket_g = std::make_tuple(g_prev, g_new.clone(at::MemoryFormat::Contiguous));
        bracket_gtd = {gtd_prev, gtd_new};
        break;
      }
    }
    // interpolate
    auto min_step = t + 0.01 * (t - t_prev);
    auto max_step = t * 10;
    auto tmp = t;
    t = _cubic_interpolate(
        t_prev,
        f_prev,
        gtd_prev,
        t,
        f_new,
        gtd_new,
        std::make_tuple(min_step, max_step));

    // next step
    t_prev = tmp;
    f_prev = f_new;
    g_prev = g_new.clone(at::MemoryFormat::Contiguous);
    gtd_prev = gtd_new;
    auto res = obj_func(x, t, d);
    f_new = std::get<0>(res);
    g_new = std::get<1>(res);
    ls_func_evals += 1;
    gtd_new = g_new.dot(d);
    ls_iter += 1;

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
    if(*bracket_f.begin() <= *(bracket_f.end() - 1)) {
      low_pos = 0;
      high_pos = 1;
    } else {
      low_pos = 1;
      high_pos = 0;
    }
    while(!done && ls_iter < max_ls) {
      // compute new trial value
      t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                             bracket[1], bracket_f[1], bracket_gtd[1]);

      // test that we are making sufficient progress:
      // in case `t` is so close to boundary, we mark that we are making
      // insufficient progress, and if
      //   + we have made insufficient progress in the last step, or
      //   + `t` is at one of the boundary,
      // we will move `t` to a position which is `0.1 * len(bracket)`
      // away from the nearest boundary point.
      double bracket_max = *std::max_element(bracket.begin(), bracket.end());
      auto bracket_min = *std::min_element(bracket.begin(), bracket.end());
      auto eps = 0.1 * (bracket_max - bracket_min);
      if (std::min(bracket_max - t, t - bracket_min) < eps) {
        // interpolation close to boundary
        if (insuf_progress || t >= bracket_max || t <= bracket_min) {
          // evaluate at 0.1 away from boundary
          t = (abs(t - bracket_max) < abs(t - bracket_min)) ? bracket_max - eps : bracket_max + eps;
          insuf_progress = false;
        } else {
          insuf_progress = true;
        }
      } else {
        insuf_progress = false;
      }
      // Evaluate new point
      res = obj_func(x, t, d);
      f_new = std::get<0>(res);
      g_new = std::get<1>(res);
      ls_func_evals += 1;
      gtd_new = g_new.dot(d);
      ls_iter += 1;

      if (f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]) {
        // Armijo condition not satisfied or not lower than lowest point
        // # Armijo condition not satisfied or not lower than lowest point
        // bracket[high_pos] = t
        // bracket_f[high_pos] = f_new
        // bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
        // bracket_gtd[high_pos] = gtd_new
        // low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
      } else {
        if (abs(gtd_new) <= -c2 * gtd) {
          // Wolfe conditions satisfied
          done = true;
        } else if (gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0) {
          // old high becomes new low
          // bracket[high_pos] = bracket[low_pos]
          // bracket_f[high_pos] = bracket_f[low_pos]
          // bracket_g[high_pos] = bracket_g[low_pos]
          // bracket_gtd[high_pos] = bracket_gtd[low_pos]
        }
        // # new point becomes new low
        //     bracket[low_pos] = t
        //     bracket_f[low_pos] = f_new
        //     bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
        //     bracket_gtd[low_pos] = gtd_new

      }
      // # line-search bracket is so small
      //   if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
      //       break
    }
    // # return stuff
    //   t = bracket[low_pos]
    //   f_new = bracket_f[low_pos]
    //   g_new = bracket_g[low_pos]
    //   return f_new, g_new, t, ls_func_evals
}

Tensor LBFGS::step(LossClosure closure) {
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
  auto closure_ = [](const LossClosure& closure) {
    torch::AutoGradMode enable_grad(true);
    return closure();
  };

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
