#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
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

struct TORCH_API SGDOptions : public OptimizerCloneableOptions<SGDOptions> {
  /* implicit */ SGDOptions(double lr);
  TORCH_ARG(double, lr);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;
public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(const SGDOptions& lhs, const SGDOptions& rhs);
  ~SGDOptions() = default;
};

struct TORCH_API SGDParamState : public OptimizerCloneableParamState<SGDParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);

public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(const SGDParamState& lhs, const SGDParamState& rhs);
  ~SGDParamState() = default;
};

class TORCH_API SGD : public Optimizer {
 public:
  explicit SGD(std::vector<OptimizerParamGroup> param_groups,
      SGDOptions defaults) : Optimizer(std::move(param_groups), std::make_unique<SGDOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.momentum() >= 0, "Invalid momentum value: ", defaults.momentum());
    TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
    TORCH_CHECK(!defaults.nesterov() || (defaults.momentum() > 0 && defaults.dampening() == 0), "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params,
      SGDOptions defaults) : SGD({std::move(OptimizerParamGroup(params))}, defaults) {}

  torch::Tensor step(LossClosure closure = nullptr) override;

  /// Adds the given vector of parameters to the optimizer's parameter list.
  void add_parameters(const std::vector<Tensor>& parameters) override;

  /// Provides a const reference to the parameters this optimizer holds.
  const std::vector<Tensor>& parameters() const noexcept override;

  /// Provides a reference to the parameters this optimizer holds.
  std::vector<Tensor>& parameters() noexcept override;

  /// Returns the number of parameters referenced by the optimizer.
  size_t size() const noexcept override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);
  }
};
} // namespace optim
} // namespace torch
