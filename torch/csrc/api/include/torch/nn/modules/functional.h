#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>

namespace torch {
namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  explicit FunctionalImpl(std::function<Tensor(Tensor)> function);

  void reset() override;
  Tensor forward(Tensor input);

  /// Calls forward(input).
  Tensor operator()(Tensor input);

 private:
  std::function<Tensor(Tensor)> function_;
};

TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
