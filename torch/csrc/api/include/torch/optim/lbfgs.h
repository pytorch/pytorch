#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>

#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace torch::optim {

struct TORCH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  // Field IDs for tracking
  static constexpr size_t LR_ID = 0;
  static constexpr size_t MAX_ITER_ID = 1;
  static constexpr size_t MAX_EVAL_ID = 2;
  static constexpr size_t TOLERANCE_GRAD_ID = 3;
  static constexpr size_t TOLERANCE_CHANGE_ID = 4;
  static constexpr size_t HISTORY_SIZE_ID = 5;
  static constexpr size_t LINE_SEARCH_FN_ID = 6;

  LBFGSOptions(double lr = 1);
  TORCH_ARG_WITH_TRACKING(double, lr, LR_ID) = 1;
  TORCH_ARG_WITH_TRACKING(int64_t, max_iter, MAX_ITER_ID) = 20;
  TORCH_ARG_WITH_TRACKING(std::optional<int64_t>, max_eval, MAX_EVAL_ID) =
      std::nullopt;
  TORCH_ARG_WITH_TRACKING(double, tolerance_grad, TOLERANCE_GRAD_ID) = 1e-7;
  TORCH_ARG_WITH_TRACKING(double, tolerance_change, TOLERANCE_CHANGE_ID) = 1e-9;
  TORCH_ARG_WITH_TRACKING(int64_t, history_size, HISTORY_SIZE_ID) = 100;
  TORCH_ARG_WITH_TRACKING(
      std::optional<std::string>,
      line_search_fn,
      LINE_SEARCH_FN_ID) = std::nullopt;

 public:
  static void merge_impl(LBFGSOptions* dst, const LBFGSOptions& src) {
    if (src.is_field_explicitly_set(LR_ID))
      dst->lr_ = src.lr_;
    if (src.is_field_explicitly_set(MAX_ITER_ID))
      dst->max_iter_ = src.max_iter_;
    if (src.is_field_explicitly_set(MAX_EVAL_ID))
      dst->max_eval_ = src.max_eval_;
    if (src.is_field_explicitly_set(TOLERANCE_GRAD_ID))
      dst->tolerance_grad_ = src.tolerance_grad_;
    if (src.is_field_explicitly_set(TOLERANCE_CHANGE_ID))
      dst->tolerance_change_ = src.tolerance_change_;
    if (src.is_field_explicitly_set(HISTORY_SIZE_ID))
      dst->history_size_ = src.history_size_;
    if (src.is_field_explicitly_set(LINE_SEARCH_FN_ID))
      dst->line_search_fn_ = src.line_search_fn_;
  }
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
  TORCH_ARG(std::optional<std::vector<Tensor>>, al) = std::nullopt;

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
      const std::vector<OptimizerParamGroup>& param_groups,
      LBFGSOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<LBFGSOptions>(defaults)) {
    TORCH_CHECK(
        param_groups_.size() == 1,
        "LBFGS doesn't support per-parameter options (parameter groups)");
    if (defaults.max_eval() == std::nullopt) {
      auto max_eval_val = (defaults.max_iter() * 5) / 4;
      static_cast<LBFGSOptions&>(param_groups_[0].options())
          .max_eval(max_eval_val);
      static_cast<LBFGSOptions&>(*defaults_).max_eval(max_eval_val);
    }
    _numel_cache = std::nullopt;
  }
  explicit LBFGS(std::vector<Tensor> params, LBFGSOptions defaults = {})
      : LBFGS({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  Tensor step(LossClosure closure) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  std::optional<int64_t> _numel_cache;
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
} // namespace torch::optim
