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

class TORCH_API OptimizerParamState {
 public:
  virtual std::unique_ptr<OptimizerParamState> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerParamState() = default;
};

template <typename Derived>
class TORCH_API OptimizerCloneableParamState : public OptimizerParamState {
  std::unique_ptr<OptimizerParamState> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

class TORCH_API OptimizerOptions {
 public:
  virtual std::unique_ptr<OptimizerOptions> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerOptions() = default;
};

template <typename Derived>
class TORCH_API OptimizerCloneableOptions : public OptimizerOptions {
  std::unique_ptr<OptimizerOptions> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

/// Stores parameters in the param_group and stores a pointer to the OptimizerOptions
class TORCH_API OptimizerParamGroup {
 public:
  // NOTE: In order to store `OptimizerParamGroup` in a `std::vector`, it has to be copy-constructible.
  OptimizerParamGroup(const OptimizerParamGroup& param_group) : params_(param_group.params()), options_(param_group.has_options() ? param_group.options().clone() : nullptr) {}
  OptimizerParamGroup(std::vector<Tensor> params) : params_(std::move(params)) {}
  OptimizerParamGroup(std::vector<Tensor> params, std::unique_ptr<OptimizerOptions> options) : params_(std::move(params)), options_(std::move(options)) {}

  bool has_options() const;
  OptimizerOptions& options();
  const OptimizerOptions& options() const;
  void set_options(std::unique_ptr<OptimizerOptions> options);
  std::vector<Tensor>& params();
  const std::vector<Tensor>& params() const;

 protected:
  std::vector<Tensor> params_;
  std::unique_ptr<OptimizerOptions> options_;
};

namespace detail {

/// Base class for all optimizers, that does not yet define a `step()`
/// mechanism. All it specifies is that optimizers must be supplied with a
/// vector of parameters. It also defines certain methods that all optimizers
/// shall have, such as `zero_grad`.
class TORCH_API OptimizerBase {
 public:
  // The copy constructor is deleted, because the user should use the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  OptimizerBase(const OptimizerBase& optimizer_base) = delete;
  OptimizerBase(OptimizerBase&& optimizer_base) = default;

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit OptimizerBase(std::vector<Tensor> parameters);

  explicit OptimizerBase(std::vector<OptimizerParamGroup> param_groups, std::unique_ptr<OptimizerOptions> defaults) : defaults_(std::move(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  /// Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const OptimizerParamGroup& param_group);

  virtual ~OptimizerBase() = default;

  // TODO: when all optimizers use the new design, we can devirtualize some of the following methods
  // such as add_parameters() / parameters() / size()

  /// Adds the given vector of parameters to the optimizer's parameter list.
  virtual void add_parameters(const std::vector<Tensor>& parameters);

  virtual void _add_parameters_new_design(const std::vector<Tensor>& parameters);

  /// Zeros out the gradients of all parameters.
  virtual void zero_grad();

  /// Provides a const reference to the parameters this optimizer holds.
  virtual const std::vector<Tensor>& parameters() const noexcept;

  virtual const std::vector<Tensor>& _parameters_new_design() const noexcept;

  /// Provides a reference to the parameters this optimizer holds.
  virtual std::vector<Tensor>& parameters() noexcept;

  virtual std::vector<Tensor>& _parameters_new_design() noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  virtual size_t size() const noexcept;

  virtual size_t _size_new_design() const noexcept;

  OptimizerOptions& defaults() noexcept;

  const OptimizerOptions& defaults() const noexcept;

  /// Provides a reference to the param_groups this optimizer holds.
  std::vector<OptimizerParamGroup>& param_groups() noexcept;

  /// Provides a const reference to the param_groups this optimizer holds.
  const std::vector<OptimizerParamGroup>& param_groups() const noexcept;

  /// Provides a reference to the state this optimizer holds
  ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state() noexcept;

  /// Provides a const reference to the state this optimizer holds
  const ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state() const noexcept;

  /// Serializes the optimizer state into the given `archive`.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the optimizer state from the given `archive`.
  virtual void load(serialize::InputArchive& archive);

 protected:
   std::vector<OptimizerParamGroup> param_groups_;
   ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>> state_;
   std::unique_ptr<OptimizerOptions> defaults_;
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
};

/* How do we decide whether to serialize undefined tensors or
  c10::nullopt values into the output archive?
Answer: we strictly follow the behavior of Python API. To be more specific:

For optimizer options:
a) For undefined tensor: currently no tensor is used as an options argument in Python API,
   so we don't need to worry about it now.
b) For c10::nullopt value: we serialize c10::nullopt values into the output archive,
   to follow the exact same behavior as Python API.

For optimizer param state:
a) For undefined tensor: in param state, undefined tensor in C++ impl is equivalent to
   missing key in Python impl. Since we don't serialize missing keys in Python API,
   we skip undefined tensors when serializing the param state.
b) For c10::nullopt value: in param state, c10::nullopt value in C++ impl is equivalent to
   missing key in Python impl. Since we don't serialize missing keys in Python API,
   we skip c10::nullopt values when serializing the param state. */

/// Serializes an `OptimizerBase` into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const OptimizerBase& optimizer);

/// Deserializes a `Tensor` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    OptimizerBase& optimizer);
} // namespace detail

/// Optimizer that can optionally take a loss function in `step()` method
/// and returns the loss value. The only side effect is that parameters are updated
/// according to the concrete optimization algorithm.
class Optimizer : public detail::OptimizerBase {
 public:
   /// A loss function closure, which is expected to return the loss value.
   using LossClosure = std::function<Tensor()>;
   using detail::OptimizerBase::OptimizerBase;
   virtual Tensor step(LossClosure closure = nullptr) = 0;
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
