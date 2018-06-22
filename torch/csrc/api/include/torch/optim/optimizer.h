#pragma once

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/nn/cursor.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace torch {
namespace optim {
namespace detail {

template <typename ParameterContainer>
std::vector<Variable> zeros_like(const ParameterContainer& parameters) {
  std::vector<Variable> result;
  result.reserve(parameters.size());
  for (auto& parameter : parameters) {
    result.push_back(torch::zeros_like(parameter));
  }
  return result;
}

class OptimizerBase {
 public:
  using ParameterCursor = torch::detail::CursorBase<Variable>;

  explicit OptimizerBase(std::vector<Variable>&& parameters)
      : parameters_(std::move(parameters)) {}

  explicit OptimizerBase(ParameterCursor&& cursor) {
    parameters_.reserve(cursor.size());
    for (auto& parameter : cursor) {
      parameters_.push_back(*parameter);
    }
  }

  virtual ~OptimizerBase() = default;

  virtual void zero_grad();

 protected:
  OptimizerBase() = default;

  std::vector<Variable> parameters_;
};
} // namespace detail

class Optimizer : public detail::OptimizerBase {
 public:
  using detail::OptimizerBase::OptimizerBase;
  virtual void step() = 0;
};

class LossClosureOptimizer : public detail::OptimizerBase {
 public:
  using LossClosure = std::function<at::Scalar()>;
  using detail::OptimizerBase::OptimizerBase;
  virtual at::Scalar step(LossClosure closure) = 0;
};

} // namespace optim
} // namespace torch
