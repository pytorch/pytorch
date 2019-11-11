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

struct TORCH_API AdagradOptions : public OptimizerCloneableOptions<AdagradOptions> {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;
};

struct TORCH_API AdagradParamState : public OptimizerCloneableParamState<AdagradParamState> {
  TORCH_ARG(torch::Tensor, sum);
  TORCH_ARG(int64_t, step);
};

class TORCH_API Adagrad : public Optimizer {
 public:
  explicit Adagrad(std::vector<OptimizerParamGroup> param_groups,
      AdagradOptions defaults) : Optimizer(std::move(param_groups), c10::guts::make_unique<AdagradOptions>(std::move(defaults))) {
    AdagradOptions* default_derived = static_cast<AdagradOptions*>(defaults_.get());
    TORCH_CHECK(default_derived->learning_rate() >= 0, "Invalid learning rate: ", default_derived->learning_rate());
    TORCH_CHECK(default_derived->lr_decay() >= 0, "Invalid lr_decay value: ", default_derived->lr_decay());
    TORCH_CHECK(default_derived->weight_decay() >= 0, "Invalid weight_decay value: ", default_derived->weight_decay());
    TORCH_CHECK(default_derived->initial_accumulator_value() >= 0, "Invalid initial_accumulator_value value: ", default_derived->initial_accumulator_value());
    TORCH_CHECK(default_derived->eps() >= 0, "Invalid epsilon value: ", default_derived->eps());

    for (const auto& group : param_groups_) {
      for (const auto& p : group.params()) {
        auto state = c10::guts::make_unique<AdagradParamState>();
        state->step(0);
        state->sum(torch::full_like(p.data(), defaults.initial_accumulator_value()));
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }
    }
  }

  // TODO: we might want to replace `std::vector<Tensor>` with `ParameterContainer` at some point
  explicit Adagrad(
      std::vector<Tensor> params,
      AdagradOptions defaults) : Adagrad({OptimizerParamGroup(params)}, defaults) {}

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
  Adagrad() : Optimizer({}, c10::guts::make_unique<AdagradOptions>(0)) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //_TORCH_OPTIM_SERIALIZE(state); add a serialize function
  }
};
} // namespace optim
} // namespace torch
