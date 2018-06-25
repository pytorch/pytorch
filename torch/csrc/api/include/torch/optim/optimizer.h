#pragma once

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/cursor.h>
#include <torch/tensor.h>

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
  explicit OptimizerBase(std::vector<Tensor> parameters)
      : parameters_(std::move(parameters)) {}

  /// Constructs the `Optimizer` from a ParameterCursor, such as
  /// `nn::Module::parameters()` returns.
  explicit OptimizerBase(ParameterCursor cursor) {
    parameters_.reserve(cursor.size());
    for (const auto& parameter : cursor) {
      parameters_.push_back(*parameter);
    }
  }

  virtual ~OptimizerBase() = default;

  /// Zeros out the gradients of all parameters.
  virtual void zero_grad();

  /// Provides a reference to the parameters this optimizer holds.
  const std::vector<Tensor>& parameters() const noexcept;

 protected:
  OptimizerBase() = default;

  /// Lazily creates a buffer (e.g. for momentum) the first time it would be
  /// accessed (when the index is equal to the size of the given buffer vector).
  /// Laziness is important because we want the buffer to have the same dtype,
  /// device, layout etc. as the corresponding parameter.
  Tensor& lazily_create_buffer(
      std::vector<Tensor>& buffers,
      size_t index,
      const Tensor& parameter);

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
