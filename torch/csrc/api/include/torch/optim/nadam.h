#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

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

struct TORCH_API NAdamOptions : public OptimizerCloneableOptions<NAdamOptions> {
  NAdamOptions(double lr = 2e-3);
  TORCH_ARG(double, lr) = 2e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum_decay) = 4e-3;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const NAdamOptions& lhs,
      const NAdamOptions& rhs);
  ~NAdamOptions() override = default;
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API NAdamParamState
    : public OptimizerCloneableParamState<NAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};
  TORCH_ARG(double, mu_product) = 1;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const NAdamParamState& lhs,
      const NAdamParamState& rhs);
  ~NAdamParamState() override = default;
};

class TORCH_API NAdam : public Optimizer {
 public:
  explicit NAdam(
      std::vector<OptimizerParamGroup> param_groups,
      NAdamOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<NAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        defaults.momentum_decay() >= 0,
        "Invalid momentum_decay value: ",
        defaults.momentum_decay());
  }
  explicit NAdam(std::vector<Tensor> params, NAdamOptions defaults = {})
      : NAdam({OptimizerParamGroup(std::move(params))}, defaults) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(NAdam);
  }
};
} // namespace optim
} // namespace torch
