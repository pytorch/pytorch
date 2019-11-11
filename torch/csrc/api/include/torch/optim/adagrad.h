#pragma once

#include <c10/util/Optional.h>
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

struct TORCH_API AdagradParamGroup {
  AdagradParamGroup(std::vector<Tensor> params) : params_(params) {}
  AdagradParamGroup(std::vector<Tensor> params, AdagradOptions options) : params_(params), options_(options) {}

  bool has_options() const {
    return options_.has_value();
  }

  const AdagradOptions& options() const {
    TORCH_CHECK(options_.has_value());
    return options_.value();
  }

  void set_options(AdagradOptions options) {
    options_ = options;
  }

  std::vector<Tensor>& params() {
    return params_;
  }

  const std::vector<Tensor>& params() const {
    return params_;
  }
 private:
  std::vector<Tensor> params_;
  c10::optional<AdagradOptions> options_;
};

struct TORCH_API AdagradParamState {
  // TODO: maybe better encapsulation
  torch::Tensor sum_;
  int64_t step_;
};

class TORCH_API Adagrad : public Optimizer<AdagradParamState, AdagradParamGroup, AdagradOptions> {
 public:
  explicit Adagrad(std::vector<AdagradParamGroup> param_groups,
      AdagradOptions defaults) : Optimizer<AdagradParamState, AdagradParamGroup, AdagradOptions>(param_groups, std::move(defaults)) {
      TORCH_CHECK(defaults_.value().learning_rate() >= 0, "Invalid learning rate: ", defaults_.value().learning_rate());
      TORCH_CHECK(defaults_.value().lr_decay() >= 0, "Invalid lr_decay value: ", defaults_.value().lr_decay());
      TORCH_CHECK(defaults_.value().weight_decay() >= 0, "Invalid weight_decay value: ", defaults_.value().weight_decay());
      TORCH_CHECK(defaults_.value().initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", defaults_.value().initial_accumulator_value());
      TORCH_CHECK(defaults_.value().eps() >= 0, "Invalid epsilon value: ", defaults_.value().eps());
  }

  // TODO: we might want to replace `std::vector<Tensor>` with `ParameterContainer` at some point
  explicit Adagrad(
      std::vector<Tensor> params,
      AdagradOptions defaults) : Adagrad({AdagradParamGroup(params)}, std::move(defaults)) {}

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
  Adagrad() : Optimizer<AdagradParamState, AdagradParamGroup, AdagradOptions>({}, std::move(AdagradOptions(0))) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //_TORCH_OPTIM_SERIALIZE(state); add a serialize function
  }
};
} // namespace optim
} // namespace torch
