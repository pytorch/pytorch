#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace mobile {

class SGDParamState {
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  std::unique_ptr<SGDParamState> clone() const {
    return std::make_unique<SGDParamState>(
        static_cast<const SGDParamState&>(*this));
  }
  friend bool operator==(const SGDParamState& lhs, const SGDParamState& rhs);
};

struct TORCH_API SGDOptions {
  /* implicit */ SGDOptions(double lr);
  TORCH_ARG(double, lr);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;

 public:
  std::unique_ptr<SGDOptions> clone() const {
    return std::make_unique<SGDOptions>(static_cast<const SGDOptions&>(*this));
  }
  TORCH_API friend bool operator==(
      const SGDOptions& lhs,
      const SGDOptions& rhs);
};

/// Stores parameters in the param_group and stores a pointer to the SGDOptions
class TORCH_API SGDParamGroup {
 public:
  // NOTE: In order to store `SGDParamGroup` in a `std::vector`, it has to be
  // copy-constructible.
  SGDParamGroup(const SGDParamGroup& param_group)
      : params_(param_group.params()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  SGDParamGroup& operator=(const SGDParamGroup& param_group) {
    this->params_ = param_group.params();
    this->options_ =
        param_group.has_options() ? param_group.options().clone() : nullptr;
    return *this;
  }
  /* implicit */ SGDParamGroup(std::vector<Tensor> params)
      : params_(std::move(params)) {}
  SGDParamGroup(std::vector<Tensor> params, std::unique_ptr<SGDOptions> options)
      : params_(std::move(params)), options_(std::move(options)) {}

  bool has_options() const;
  SGDOptions& options();
  const SGDOptions& options() const;
  void set_options(std::unique_ptr<SGDOptions> options);
  std::vector<Tensor>& params();
  const std::vector<Tensor>& params() const;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> options_;
};

class TORCH_API SGD {
 public:
  explicit SGD(
      const std::vector<torch::jit::mobile::SGDParamGroup>& param_groups,
      SGDOptions defaults)
      : defaults_(std::make_unique<SGDOptions>(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({SGDParamGroup(std::move(params))}, defaults) {}

  /// Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const SGDParamGroup& param_group);

  ~SGD() = default;

  using LossClosure = std::function<Tensor()>;
  /// A loss function closure, which is expected to return the loss value.
  torch::Tensor step(const LossClosure& closure = nullptr);

  /// Zeros out the gradients of all parameters.
  void zero_grad();

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<SGDParamGroup> param_groups_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ska::flat_hash_map<void*, std::unique_ptr<SGDParamState>> state_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> defaults_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> options_;
};
} // namespace mobile
} // namespace jit
} // namespace torch
