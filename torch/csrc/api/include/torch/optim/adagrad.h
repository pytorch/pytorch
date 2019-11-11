#pragma once

#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
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

struct TORCH_API AdagradOptions : public detail::OptimizerOptionsBase {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;
};

struct TORCH_API AdagradParamState : public detail::OptimizerParamStateBase {
  // TODO: maybe better encapsulation
  torch::Tensor sum_;
  int64_t step_;
};

class TORCH_API Adagrad : public Optimizer {
 public:
  explicit Adagrad(std::vector<detail::OptimizerParamGroup> param_groups,
      AdagradOptions defaults) : Optimizer(param_groups, std::make_shared<AdagradOptions>(defaults)) {
    TORCH_CHECK(defaults.learning_rate() >= 0, "Invalid learning rate: ", defaults.learning_rate());
    TORCH_CHECK(defaults.lr_decay() >= 0, "Invalid lr_decay value: ", defaults.lr_decay());
    TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
    TORCH_CHECK(defaults.initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", defaults.initial_accumulator_value());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());

    for (const auto& group : param_groups) {
      for (const auto& p : group.params()) {
        auto state = std::make_shared<AdagradParamState>();
        state->step_ = 0;
        state->sum_ = torch::full_like(p.data(), defaults.initial_accumulator_value());
        state_[p.unsafeGetTensorImpl()] = state;
      }
    }
  }

  // TODO: we might want to replace `std::vector<Tensor>` with `ParameterContainer` at some point
  explicit Adagrad(
      std::vector<Tensor> params,
      AdagradOptions defaults) : Adagrad({detail::OptimizerParamGroup(params)}, defaults) {}

  void step() override;

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
  Adagrad() : Optimizer({}, std::make_shared<AdagradOptions>(0)) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //_TORCH_OPTIM_SERIALIZE(state); add a serialize function
  }
};
} // namespace optim
} // namespace torch
