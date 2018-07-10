#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
struct LinearOptions {
  LinearOptions(int64_t in, int64_t out);
  TORCH_ARG(int64_t, in);
  TORCH_ARG(int64_t, out);
  TORCH_ARG(bool, with_bias) = true;
};

class LinearImpl : public Cloneable<LinearImpl> {
 public:
  explicit LinearImpl(LinearOptions options);

  void reset() override;
  Tensor forward(Tensor);

  LinearOptions options;
  Tensor weight;
  Tensor bias;
};

TORCH_MODULE(Linear);

} // namespace nn
} // namespace torch
