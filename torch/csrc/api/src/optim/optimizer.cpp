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
  std::vector<Tensor> tensors(cursor.size());
  cursor.map(tensors.begin(), [](const Tensor& tensor) { return tensor; });
  add_parameters(tensors);
}

void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    if (parameter.grad().defined()) {
      parameter.grad().detach_();
      parameter.grad().zero_();
    }
  }
}

size_t OptimizerBase::size() const noexcept {
  return parameters_.size();
}
} // namespace detail
} // namespace optim
} // namespace torch
