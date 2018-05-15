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

  TORCH_PARAMETER(int64_t, count);
  TORCH_PARAMETER(int64_t, dimension);

 private:
  Variable table_;
};
}} // namespace torch::nn
