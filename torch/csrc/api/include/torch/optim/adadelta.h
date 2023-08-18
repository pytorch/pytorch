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

struct TORCH_API AdadeltaOptions
    : public OptimizerCloneableOptions<AdadeltaOptions> {
  AdadeltaOptions(double lr = 1.0);
  TORCH_ARG(double, lr) = 1.0;
  TORCH_ARG(double, rho) = 0.9;
  TORCH_ARG(double, eps) = 1e-6;
  TORCH_ARG(double, weight_decay) = 0;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdadeltaOptions& lhs,
      const AdadeltaOptions& rhs);
  ~AdadeltaOptions() override = default;
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API AdadeltaParamState 
    : public OptimizerCloneableParamState<AdadeltaParamState> {
  TORCH_ARG(torch::Tensor, square_avg);
  TORCH_ARG(torch::Tensor, accumulate);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdadeltaParamState& lhs,
      const AdadeltaParamState& rhs);
  ~AdadeltaParamState() override = default;
};

class TORCH_API Adadelta : public Optimizer {
 public:
  explicit Adadelta(
      std::vector<OptimizerParamGroup> param_groups,
      AdadeltaOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<AdadeltaOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.rho() >= 0,
        "Invalid rho value: ",
        defaults.rho());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
  }

  explicit Adadelta(std::vector<Tensor> params, AdadeltaOptions defaults = {})
      : Adadelta({OptimizerParamGroup(std::move(params))}, defaults) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adadelta);
  }
};
} // namespace optim
} // namespace torch
