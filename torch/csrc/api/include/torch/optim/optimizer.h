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
  using ParameterCursor = torch::detail::CursorBase<Variable>;

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit OptimizerBase(std::vector<Variable> parameters)
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
  const std::vector<Variable>& parameters() const noexcept;

 protected:
  OptimizerBase() = default;

  /// The parameters this optimizer optimizes.
  std::vector<Variable> parameters_;
};

/// Helper function to construct a vector of zero-d out variables, each the same
/// shape as the variable at the corresponding index in the input container.
template <typename ParameterContainer>
std::vector<Variable> zeros_like(const ParameterContainer& parameters) {
  std::vector<Variable> result;
  result.reserve(parameters.size());
  for (auto& parameter : parameters) {
    result.push_back(torch::zeros_like(parameter));
  }
  return result;
}
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
  using LossClosure = std::function<Variable()>;
  using detail::OptimizerBase::OptimizerBase;
  virtual Variable step(LossClosure closure) = 0;
};

} // namespace optim
} // namespace torch
