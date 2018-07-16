#include <torch/optim/optimizer.h>

#include <torch/nn/cursor.h>
#include <torch/tensor.h>

#include <utility>
#include <vector>

namespace torch {
namespace optim {
namespace detail {

OptimizerBase::OptimizerBase(std::vector<Tensor> parameters)
    : parameters_(std::move(parameters)) {}

OptimizerBase::OptimizerBase(const ParameterCursor& cursor) {
  add_parameters(cursor);
}

void OptimizerBase::add_parameters(const std::vector<Tensor>& parameters) {
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void OptimizerBase::add_parameters(const ParameterCursor& cursor) {
  parameters_.reserve(parameters_.size() + cursor.size());
  for (const auto& parameter : cursor) {
    parameters_.push_back(*parameter);
  }
}

void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    auto& grad = parameter.grad();
    if (grad.defined()) {
      grad = grad.detach();
      Tensor(grad).data().zero_();
    }
  }
}

size_t OptimizerBase::size() const noexcept {
  return parameters_.size();
}
} // namespace detail
} // namespace optim
} // namespace torch
