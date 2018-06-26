#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <vector>

namespace torch {
namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<std::vector<Tensor>(std::vector<Tensor>)>;

  explicit FunctionalImpl(Function function);
  explicit FunctionalImpl(std::function<Tensor(Tensor)> function);

  void reset() override;
  std::vector<Tensor> forward(std::vector<Tensor> input);

 private:
  Function function_;
};

TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
