#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Optim : public torch::lazy::TsNode {
 public:
  Optim(const torch::lazy::Value& input, const torch::lazy::Value& grad);

  std::string ToString() const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
