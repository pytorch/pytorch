#pragma once

#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API AdagradOptions
    : public OptimizerCloneableOptions<AdagradOptions> {
  AdagradOptions(double lr = 1e-2);
  TORCH_ARG(double, lr) = 1e-2;
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdagradOptions& lhs,
      const AdagradOptions& rhs);
  ~AdagradOptions() override = default;
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API AdagradParamState
    : public OptimizerCloneableParamState<AdagradParamState> {
  TORCH_ARG(torch::Tensor, sum);
  TORCH_ARG(int64_t, step) = 0;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdagradParamState& lhs,
      const AdagradParamState& rhs);
  ~AdagradParamState() override = default;
};

class TORCH_API Adagrad : public Optimizer {
 public:
  explicit Adagrad(
      std::vector<OptimizerParamGroup> param_groups,
      AdagradOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<AdagradOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.lr_decay() >= 0,
        "Invalid lr_decay value: ",
        defaults.lr_decay());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        defaults.initial_accumulator_value() >= 0,
        "Invalid initial_accumulator_value value: ",
        defaults.initial_accumulator_value());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());

    for (const auto& group : param_groups_) {
      for (const auto& p : group.params()) {
        auto state = std::make_unique<AdagradParamState>();
        state->step(0);
        state->sum(torch::full_like(
            p.data(),
            defaults.initial_accumulator_value(),
            at::MemoryFormat::Preserve));
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] =
            std::move(state);
      }
    }
  }

  explicit Adagrad(std::vector<Tensor> params, AdagradOptions defaults = {})
      : Adagrad({OptimizerParamGroup(std::move(params))}, defaults) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad);
  }
};
} // namespace optim
} // namespace torch
