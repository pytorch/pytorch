#pragma once

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/cursor.h>
#include <torch/tensor.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace optim {
namespace detail {

/// Base class for all optimizers, that does not yet define a `step()`
/// mechanism. All it specifies is that optimizers must be supplied with a
/// vector of parameters. It also defines certain methods that all optimizers
/// shall have, such as `zero_grad`.
class OptimizerBase {
 public:
  using ParameterCursor = torch::detail::CursorBase<Tensor>;

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit OptimizerBase(std::vector<Tensor> parameters);

  /// Constructs the `Optimizer` from a ParameterCursor, such as
  /// `nn::Module::parameters()` returns.
  explicit OptimizerBase(const ParameterCursor& cursor);

  virtual ~OptimizerBase() = default;

  /// Adds the given vector of parameters to the optimizer's parameter list.
  /// Override this method if you want to modify the way parameters are added to
  /// the `Optimizer`.
  virtual void add_parameters(const std::vector<Tensor>& parameters);

  /// Adds the `ParameterCursor`'s parameters to the optimizer's parameter list.
  /// NOTE: Calls the `vector<Tensor>` overload of `add_parameters` -- override
  /// that method if you want to modify the behavior of `add_parameters`.
  virtual void add_parameters(const ParameterCursor& cursor);

  /// Zeros out the gradients of all parameters.
  virtual void zero_grad();

  /// Provides a reference to the parameters this optimizer holds.
  const std::vector<Tensor>& parameters() const noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  size_t size() const noexcept;

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
  Tensor& buffer_at(std::vector<Tensor>& buffers, size_t index) {
    if (buffers.size() <= index) {
      buffers.reserve(index);
      for (auto i = buffers.size(); i <= index; ++i) {
        buffers.push_back(torch::zeros_like(parameters_.at(i)));
      }
    }
    // Copy the buffer to the device and dtype of the parameter.
    const auto& parameter = parameters_.at(index);
    const auto& buffer = buffers.at(index);
    if (buffer.device() != parameter.device() ||
        buffer.dtype() != parameter.dtype()) {
      buffers[index] = buffer.to(parameter.device(), parameter.dtype());
    }
    return buffers[index];
  }

  /// The parameters this optimizer optimizes.
  std::vector<Tensor> parameters_;
};
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
