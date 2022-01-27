#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include "ATen/core/ivalue.h"

namespace torch {
namespace lazy {

class TORCH_API FunctionCall : public TsNode {
 public:
  FunctionCall(torch::lazy::OpList values, c10::ArrayRef<c10::IValue> consts, torch::jit::Function* f, std::vector<Shape> shapes);

  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  std::vector<c10::IValue> consts_;
  torch::jit::Function* function_;
};

}  // namespace lazy
}  // namespace torch

