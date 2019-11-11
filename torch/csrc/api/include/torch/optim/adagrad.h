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

public:
  at::IValue convert_options_to_ivalue() {
      c10::impl::GenericDict dict(c10::StringType::get(), c10::AnyType::get());
      dict.insert("learning_rate", learning_rate());
      dict.insert("lr_decay", lr_decay());
      dict.insert("weight_decay", weight_decay());
      dict.insert("initial_accumulator_value", initial_accumulator_value());
      dict.insert("eps", eps());

      at::IValue ivalue = dict;
      return ivalue;
  }
};

class TORCH_API Adagrad : public Optimizer {
 public:
  // explicit Adagrad(
  //     ParameterContainer&& parameters,
  //     const AdagradOptions& options_)
  //     : Optimizer(std::forward<ParameterContainer>(parameters)),
  //       options(options_) {}
  //template <typename ParameterContainer>
  explicit Adagrad(
        std::vector<std::vector<Tensor>> parameters,
        const AdagradOptions& options_) : Optimizer(std::forward<std::vector<std::vector<Tensor>>>(parameters), options_), options(options_) {
      TORCH_CHECK(options.learning_rate() >= 0, "Invalid learning rate: ", options.learning_rate());
      TORCH_CHECK(options.lr_decay() >= 0, "Invalid lr_decay value: ", options.lr_decay());
      TORCH_CHECK(options.weight_decay() >= 0, "Invalid weight_decay value: ", options.weight_decay());
      TORCH_CHECK(options.initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", options.initial_accumulator_value());
      TORCH_CHECK(options.eps() >= 0, "Invalid epsilon value: ", options.eps());
    }

  explicit Adagrad(std::vector<c10::impl::GenericDict> param_groups,
      const AdagradOptions& options_) : Optimizer(param_groups), options(options_) {
      TORCH_CHECK(options.learning_rate() >= 0, "Invalid learning rate: ", options.learning_rate());
      TORCH_CHECK(options.lr_decay() >= 0, "Invalid lr_decay value: ", options.lr_decay());
      TORCH_CHECK(options.weight_decay() >= 0, "Invalid weight_decay value: ", options.weight_decay());
      TORCH_CHECK(options.initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", options.initial_accumulator_value());
      TORCH_CHECK(options.eps() >= 0, "Invalid epsilon value: ", options.eps());
  }

  void step() override;

  AdagradOptions options;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  Adagrad() : options(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //_TORCH_OPTIM_SERIALIZE(state); add a serialize function
  }
};
} // namespace optim
} // namespace torch
