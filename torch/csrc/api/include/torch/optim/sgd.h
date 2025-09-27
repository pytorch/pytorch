#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API SGDOptions : public OptimizerCloneableOptions<SGDOptions> {
  // Field IDs for tracking
  static constexpr size_t LR_ID = 0;
  static constexpr size_t MOMENTUM_ID = 1;
  static constexpr size_t DAMPENING_ID = 2;
  static constexpr size_t WEIGHT_DECAY_ID = 3;
  static constexpr size_t NESTEROV_ID = 4;

  SGDOptions(double lr);

  TORCH_ARG_WITH_TRACKING(double, lr, LR_ID);
  TORCH_ARG_WITH_TRACKING(double, momentum, MOMENTUM_ID) = 0;
  TORCH_ARG_WITH_TRACKING(double, dampening, DAMPENING_ID) = 0;
  TORCH_ARG_WITH_TRACKING(double, weight_decay, WEIGHT_DECAY_ID) = 0;
  TORCH_ARG_WITH_TRACKING(bool, nesterov, NESTEROV_ID) = false;

 public:
  static void merge_impl(SGDOptions* dst, const SGDOptions& src) {
    if (src.is_field_explicitly_set(LR_ID))
      dst->lr_ = src.lr_;
    if (src.is_field_explicitly_set(MOMENTUM_ID))
      dst->momentum_ = src.momentum_;
    if (src.is_field_explicitly_set(DAMPENING_ID))
      dst->dampening_ = src.dampening_;
    if (src.is_field_explicitly_set(WEIGHT_DECAY_ID))
      dst->weight_decay_ = src.weight_decay_;
    if (src.is_field_explicitly_set(NESTEROV_ID))
      dst->nesterov_ = src.nesterov_;
  }
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const SGDOptions& lhs,
      const SGDOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API SGDParamState
    : public OptimizerCloneableParamState<SGDParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const SGDParamState& lhs,
      const SGDParamState& rhs);
};

class TORCH_API SGD : public Optimizer {
 public:
  explicit SGD(
      const std::vector<OptimizerParamGroup>& param_groups,
      SGDOptions defaults)
      : Optimizer(param_groups, std::make_unique<SGDOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);
  }
};
} // namespace torch::optim
