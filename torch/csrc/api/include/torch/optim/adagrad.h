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

struct TORCH_API AdagradOptions {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;
};

class TORCH_API Adagrad : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit Adagrad(
      ParameterContainer&& parameters,
      const AdagradOptions& options_) : Optimizer(std::forward<ParameterContainer>(parameters), options_), defaultOptions(options_) {
    TORCH_CHECK(defaultOptions.learning_rate() >= 0, "Invalid learning rate: ", defaultOptions.learning_rate());
    TORCH_CHECK(defaultOptions.lr_decay() >= 0, "Invalid lr_decay value: ", defaultOptions.lr_decay());
    TORCH_CHECK(defaultOptions.weight_decay() >= 0, "Invalid weight_decay value: ", defaultOptions.weight_decay());
    TORCH_CHECK(defaultOptions.initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", defaultOptions.initial_accumulator_value());
    TORCH_CHECK(defaultOptions.eps() >= 0, "Invalid epsilon value: ", defaultOptions.eps());
  }

  explicit Adagrad(std::vector<c10::Dict<std::string, at::IValue>> param_groups,
      const AdagradOptions& options_) : Optimizer(param_groups), defaultOptions(options_) {}

  void step() override;

  AdagradOptions defaultOptions;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  Adagrad() : defaultOptions(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //_TORCH_OPTIM_SERIALIZE(state); add a serialize function
  }
};
} // namespace optim
} // namespace torch
