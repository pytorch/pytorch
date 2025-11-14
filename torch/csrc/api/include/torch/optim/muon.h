#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

const double EPS = 1e-7;
const double DEFAULT_A = 3.4445;
const double DEFAULT_B = -4.7750;
const double DEFAULT_C = 2.0315;
const double DEFAULT_NS_STEPS = 5;

struct TORCH_API MuonOptions : public OptimizerCloneableOptions<MuonOptions> {
  MuonOptions(double lr = 1e-3);
  TORCH_ARG(double, lr) = 1e-3;
  TORCH_ARG(double, weight_decay) = 0.1;
  TORCH_ARG(double, momentum) = 0.95;
  TORCH_ARG(bool, nesterov) = true;
  typedef std::tuple<double, double, double> ns_coefficients_t;
  TORCH_ARG(ns_coefficients_t, ns_coefficients) = std::make_tuple(DEFAULT_A, DEFAULT_B, DEFAULT_C);
  TORCH_ARG(double, eps) = EPS;
  TORCH_ARG(int, ns_steps) = DEFAULT_NS_STEPS;
  TORCH_ARG(bool, match_rms_adamw) = true;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const MuonOptions& lhs,
      const MuonOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API MuonParamState
    : public OptimizerCloneableParamState<MuonParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const MuonParamState& lhs,
      const MuonParamState& rhs);
};

class TORCH_API Muon : public Optimizer {
 public:
  explicit Muon(
      const std::vector<OptimizerParamGroup>& param_groups,
      MuonOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<MuonOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit Muon(std::vector<Tensor> params, MuonOptions defaults = {})
      : Muon({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Muon);
  }
};
} // namespace torch::optim
