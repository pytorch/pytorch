#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class Embedding : public torch::nn::CloneableModule<Embedding> {
 public:
  Embedding(int64_t count, int64_t dimension);

  void reset() override;

  variable_list forward(variable_list) override;

  TORCH_ATTR(int64_t, count);
  TORCH_ATTR(int64_t, dimension);
  TORCH_ATTR(Variable, table);
};
}} // namespace torch::nn
