#pragma once

#include <ATen/Tensor.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/Exception.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// Forward declarations confuse Doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch {
namespace optim {
namespace detail {

class TORCH_API OptimizerParamStateBase {
 public:
  virtual std::unique_ptr<OptimizerParamStateBase> clone() const = 0;
};

class TORCH_API OptimizerOptionsBase {
 public:
  virtual std::unique_ptr<OptimizerOptionsBase> clone() const = 0;
};

class TORCH_API OptimizerParamGroup {
 public:
  // NOTE: In order to store `OptimizerParamGroup` in a `std::vector`, it has to be copy-constructible.
  OptimizerParamGroup(const OptimizerParamGroup& param_group) : params_(param_group.params()), options_(param_group.has_options() ? param_group.options()->clone() : nullptr) {}
  OptimizerParamGroup(std::vector<Tensor> params) : params_(params) {}
  OptimizerParamGroup(std::vector<Tensor> params, std::unique_ptr<OptimizerOptionsBase> options) : params_(params), options_(std::move(options)) {}

  bool has_options() const {
    return options_ != nullptr;
  }

  OptimizerOptionsBase* options() {
    TORCH_CHECK(has_options());
    return options_.get();
  }

  const OptimizerOptionsBase* options() const {
    TORCH_CHECK(has_options());
    return options_.get();
  }

  void set_options(std::unique_ptr<OptimizerOptionsBase> options) {
    options_ = std::move(options);
  }

  std::vector<Tensor>& params() {
    return params_;
  }

  const std::vector<Tensor>& params() const {
    return params_;
  }

 protected:
  std::vector<Tensor> params_;
  std::unique_ptr<OptimizerOptionsBase> options_;
};

/// Base class for all optimizers, that does not yet define a `step()`
/// mechanism. All it specifies is that optimizers must be supplied with a
/// vector of parameters. It also defines certain methods that all optimizers
/// shall have, such as `zero_grad`.
class TORCH_API OptimizerBase {
 public:
  // The copy constructor is deleted, because the user should rely on the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  OptimizerBase(const OptimizerBase& optimizer_base) = delete;
  OptimizerBase(OptimizerBase&& optimizer_base) = default;

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit OptimizerBase(std::vector<Tensor> parameters);

  //todo
  explicit OptimizerBase(std::vector<OptimizerParamGroup> param_groups, std::unique_ptr<OptimizerOptionsBase> defaults) : defaults_(std::move(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  void add_param_group(const OptimizerParamGroup& param_group) {
    for (const auto& param : param_group.params()) {
      TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
    }

    OptimizerParamGroup param_group_(param_group.params());
    if (!param_group.has_options()) {
      param_group_.set_options(defaults_->clone());
    } else {
      param_group_.set_options(param_group.options()->clone());
    }
    // TODO: check "some parameters appear in more than one parameter group"
    param_groups_.push_back(std::move(param_group_));
  }

  virtual ~OptimizerBase() = default;

  // TODO: when all optimizers use the new design, we can devirtualize some of the following methods
  // such as add_parameters() / parameters() / size()

  /// Adds the given vector of parameters to the optimizer's parameter list.
  virtual void add_parameters(const std::vector<Tensor>& parameters);

  /// Zeros out the gradients of all parameters.
  virtual void zero_grad();

  /// Provides a const reference to the parameters this optimizer holds.
  virtual const std::vector<Tensor>& parameters() const noexcept;

  /// Provides a reference to the parameters this optimizer holds.
  virtual std::vector<Tensor>& parameters() noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  virtual size_t size() const noexcept;

  /// Serializes the optimizer state into the given `archive`.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the optimizer state from the given `archive`.
  virtual void load(serialize::InputArchive& archive);

 protected:
  OptimizerBase() = default;

  /// Accesses a buffer at the given index.
  /// Additionally, zeros out the buffers when this is called on the index
  template <typename T>
  T& buffer_at(std::vector<T>& buffers, size_t index) {
    if (buffers.size() <= index) {
      const auto old_size = buffers.size();
      buffers.resize(index + 1);
      std::fill(buffers.begin() + old_size, buffers.end(), T{0});
    }
    return buffers[index];
  }

  /// Accesses a buffer at the given index, converts it to the type of the
  /// parameter at the corresponding index (a no-op if they match).
  /// Additionally, zeros out the buffers when this is called on the index
  Tensor& buffer_at(std::vector<Tensor>& buffers, size_t index);

  /// The parameters this optimizer optimizes.
  std::vector<Tensor> parameters_;
  //to do-description
  std::unique_ptr<OptimizerOptionsBase> defaults_;
  std::vector<OptimizerParamGroup> param_groups_;
  ska::flat_hash_map<at::TensorImpl*, std::unique_ptr<OptimizerParamStateBase>> state_;
};

/// Serializes an `OptimizerBase` into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const OptimizerBase& optimizer);
 
/// Deserializes a `Tensor` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    OptimizerBase& optimizer);
} // namespace detail

/// Optimizer that defines a required `step()` method that takes no arguments
/// and produces no values. The only side effect is that parameters are updated
/// according to the concrete optimization algorithm.
class Optimizer : public detail::OptimizerBase {
 public:
  using detail::OptimizerBase::OptimizerBase;
  virtual void step() = 0;
};

/// Optimizer that requires the loss function to be supplied to the `step()`
/// function, as it may evaluate the loss function multiple times per step.
/// Examples of such algorithms are conjugate gradient and LBFGS. The `step()`
/// function also returns the loss value.
class LossClosureOptimizer : public detail::OptimizerBase {
 public:
  /// A loss function closure, which is expected to return the loss value.
  using LossClosure = std::function<Tensor()>;
  using detail::OptimizerBase::OptimizerBase;
  virtual Tensor step(LossClosure closure) = 0;
};

} // namespace optim
} // namespace torch
