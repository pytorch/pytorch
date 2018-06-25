#include <torch/optim/optimizer.h>

#include <torch/tensor.h>

#include <ATen/Error.h>

namespace torch {
namespace optim {
namespace detail {
void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    auto& grad = parameter.grad();
    if (grad.defined()) {
      grad = grad.detach();
      Tensor(grad).data().zero_();
    }
  }
}

Tensor& OptimizerBase::lazily_create_buffer(
    std::vector<Tensor>& buffers,
    size_t index,
    const Tensor& parameter) {
  AT_ASSERT(index <= buffers.size());
  if (index == buffers.size()) {
    buffers.push_back(torch::zeros_like(parameter));
  }
  return buffers[index];
}

} // namespace detail
} // namespace optim
} // namespace torch
