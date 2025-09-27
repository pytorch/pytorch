#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API RMSpropOptions
    : public OptimizerCloneableOptions<RMSpropOptions> {
  // Field IDs for tracking
  static constexpr size_t LR_ID = 0;
  static constexpr size_t ALPHA_ID = 1;
  static constexpr size_t EPS_ID = 2;
  static constexpr size_t WEIGHT_DECAY_ID = 3;
  static constexpr size_t MOMENTUM_ID = 4;
  static constexpr size_t CENTERED_ID = 5;

  RMSpropOptions(double lr = 1e-2);
  TORCH_ARG_WITH_TRACKING(double, lr, LR_ID) = 1e-2;
  TORCH_ARG_WITH_TRACKING(double, alpha, ALPHA_ID) = 0.99;
  TORCH_ARG_WITH_TRACKING(double, eps, EPS_ID) = 1e-8;
  TORCH_ARG_WITH_TRACKING(double, weight_decay, WEIGHT_DECAY_ID) = 0;
  TORCH_ARG_WITH_TRACKING(double, momentum, MOMENTUM_ID) = 0;
  TORCH_ARG_WITH_TRACKING(bool, centered, CENTERED_ID) = false;

 public:
  static void merge_impl(RMSpropOptions* dst, const RMSpropOptions& src) {
    if (src.is_field_explicitly_set(LR_ID))
      dst->lr_ = src.lr_;
    if (src.is_field_explicitly_set(ALPHA_ID))
      dst->alpha_ = src.alpha_;
    if (src.is_field_explicitly_set(EPS_ID))
      dst->eps_ = src.eps_;
    if (src.is_field_explicitly_set(WEIGHT_DECAY_ID))
      dst->weight_decay_ = src.weight_decay_;
    if (src.is_field_explicitly_set(MOMENTUM_ID))
      dst->momentum_ = src.momentum_;
    if (src.is_field_explicitly_set(CENTERED_ID))
      dst->centered_ = src.centered_;
  }
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const RMSpropOptions& lhs,
      const RMSpropOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API RMSpropParamState
    : public OptimizerCloneableParamState<RMSpropParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, square_avg);
  TORCH_ARG(torch::Tensor, momentum_buffer) = {};
  TORCH_ARG(torch::Tensor, grad_avg) = {};

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const RMSpropParamState& lhs,
      const RMSpropParamState& rhs);
};

class TORCH_API RMSprop : public Optimizer {
 public:
  explicit RMSprop(
      const std::vector<OptimizerParamGroup>& param_groups,
      RMSpropOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<RMSpropOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());
  }

  explicit RMSprop(std::vector<Tensor> params, RMSpropOptions defaults = {})
      : RMSprop({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {
  }

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RMSprop);
  }
};
} // namespace torch::optim
