#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace optim {

struct TORCH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  LBFGSOptions(double lr = 1);
  TORCH_ARG(double, lr) = 1;
  TORCH_ARG(int64_t, max_iter) = 20;
  TORCH_ARG(c10::optional<int64_t>, max_eval) = c10::nullopt;
  TORCH_ARG(double, tolerance_grad) = 1e-7;
  TORCH_ARG(double, tolerance_change) = 1e-9;
  TORCH_ARG(int64_t, history_size) = 100;
  TORCH_ARG(c10::optional<std::string>, line_search_fn) = c10::nullopt;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const LBFGSOptions& lhs,
      const LBFGSOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API LBFGSParamState
    : public OptimizerCloneableParamState<LBFGSParamState> {
  TORCH_ARG(int64_t, func_evals) = 0;
  TORCH_ARG(int64_t, n_iter) = 0;
  TORCH_ARG(double, t) = 0;
  TORCH_ARG(double, prev_loss) = 0;
  TORCH_ARG(Tensor, d) = {};
  TORCH_ARG(Tensor, H_diag) = {};
  TORCH_ARG(Tensor, prev_flat_grad) = {};
  TORCH_ARG(std::deque<Tensor>, old_dirs);
  TORCH_ARG(std::deque<Tensor>, old_stps);
  TORCH_ARG(std::deque<Tensor>, ro);
  TORCH_ARG(c10::optional<std::vector<Tensor>>, al) = c10::nullopt;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const LBFGSParamState& lhs,
      const LBFGSParamState& rhs);
};

class TORCH_API LBFGS : public Optimizer {
 public:
  explicit LBFGS(
      std::vector<OptimizerParamGroup> param_groups,
      LBFGSOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<LBFGSOptions>(defaults)) {
    TORCH_CHECK(
        param_groups_.size() == 1,
        "LBFGS doesn't support per-parameter options (parameter groups)");
    if (defaults.max_eval() == c10::nullopt) {
      auto max_eval_val = (defaults.max_iter() * 5) / 4;
      static_cast<LBFGSOptions&>(param_groups_[0].options())
          .max_eval(max_eval_val);
      static_cast<LBFGSOptions&>(*defaults_.get()).max_eval(max_eval_val);
    }
    _numel_cache = c10::nullopt;
  }
  explicit LBFGS(std::vector<Tensor> params, LBFGSOptions defaults = {})
      : LBFGS({OptimizerParamGroup(std::move(params))}, defaults) {}

  Tensor step(LossClosure closure) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  c10::optional<int64_t> _numel_cache;
  int64_t _numel();
  Tensor _gather_flat_grad();
  void _add_grad(const double step_size, const Tensor& update);
  std::tuple<double, Tensor> _directional_evaluate(
      const LossClosure& closure,
      const std::vector<Tensor>& x,
      double t,
      const Tensor& d);
  void _set_param(const std::vector<Tensor>& params_data);
  std::vector<Tensor> _clone_param();

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(LBFGS);
  }
};
} // namespace optim
} // namespace torch
